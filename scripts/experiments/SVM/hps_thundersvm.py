import logging
import argparse
import pickle

import pandas as pd
import numpy as np

import scipy.io

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from utils.usp_utils import *
from thundersvm import SVC

def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'


    parser = argparse.ArgumentParser()

    
    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)

    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='turn on debug logging', required=False)
    
    args = parser.parse_args()
    data_params = {}

    lower = 1.5
    upper = 2.5
    step = 0.1

    SNRs = np.round(np.arange(lower, upper+step, step), 1)

    if args.snr_int != None:
        if len(args.snr_int) == 2:
            lower = args.snr_int[0]
            upper = args.snr_int[1]

        if len(args.snr_int) == 3:
            lower = args.snr_int[0]
            upper = args.snr_int[1]
            step = args.snr_int[2]

        SNRs = np.round(np.arange(lower, upper+step, step), 1)

        if len(args.snr_int) == 1:
            SNRs = [args.snr_int[0]]
    
    
    
    noise_dataset = args.noise_dataset.upper()
    signal_dataset = args.signal_dataset.upper()
    kernel_type = args.kernel_type
    n_folds = args.n_folds if args.n_folds != None else 7
    log_level = 'DEBUG' if args.verbose else 'INFO'

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Print the parsed arguments to verify
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Kernel Type: {kernel_type}')
    logging.info(f'Number of Folds: {n_folds}')
    

    data_params['noise_dataset'] = noise_dataset
    data_params['signal_dataset'] = signal_dataset


    signal_data = pd.read_pickle(f'{project_dir}/assets/data/{signal_dataset}_data.pkl')
    noise_data = scipy.io.loadmat(f'{project_dir}/assets/data/{noise_dataset}_data.mat')

    subjects = np.unique(signal_data['subject'])
    data_params['subjects'] = subjects


    if noise_dataset == "VAR":
        A = noise_data['A']
        u_rate = 1
        nstd = 1.0
        burn = 100
        threshold = 0.0001
        
        logging.debug(f'A - {A}')
        logging.debug(f'u_rate - {u_rate}')
        logging.debug(f'nstd - {nstd}')
        logging.debug(f'burn - {burn}')
        logging.debug(f'threshold - {threshold}')


        data_params['A'] = A
        data_params['u_rate'] = u_rate
        data_params['nstd'] = nstd
        data_params['burn'] = burn
        data_params['threshold'] = threshold

    else:
        L = noise_data['L']
        covariance_matrix = noise_data['cov_mat']

        logging.debug(f'L {L}')
        logging.debug(f'covariance_matrix {covariance_matrix}')

        data_params['L'] = L
        data_params['covariance_matrix'] = covariance_matrix


    if signal_dataset == 'OULU':
        undersampling_rate = 1
        NOISE_SIZE = 2961*2
    
    if signal_dataset == 'SIMULATION':
        undersampling_rate = 1
        NOISE_SIZE = 18018 #might should write a function to compute this, it is LCM(t1*k1, t2*k2)

    if signal_dataset == 'HCP':
        NOISE_SIZE = 1200
        undersampling_rate = 6

    
    

    data_params['NOISE_SIZE'] = NOISE_SIZE
    data_params['undersampling_rate'] = undersampling_rate

    



    for SNR in SNRs:
        data_params['SNR'] = SNR

        ################ loading and preprocessing
        all_data = load_timecourses(signal_data, data_params)



        data_df = pd.DataFrame(all_data)



        ################ windowing
        sr1_data, sr2_data, add_data, concat_data, combcov_data = perform_windowing(data_df)
        

        X_tr100, y_tr100, group_tr100 = parse_X_y_groups(pd.DataFrame(sr1_data), 'SR1')
        X_tr2150, y_tr2150, group_tr2150 = parse_X_y_groups(pd.DataFrame(sr2_data), 'SR2')
        X_add, y_add, group_add = parse_X_y_groups(pd.DataFrame(add_data), 'Add')
        X_concat, y_concat, group_concat = parse_X_y_groups(pd.DataFrame(concat_data), 'Concat')
        X_combcov, y_combcov, group_combcov = parse_X_y_groups(pd.DataFrame(combcov_data), 'CombCov')


        # Define hyperparameter grid
        C_values = [0.1, 1.0, 10.0]
        coef0_values = [0.0, 0.1, 1.0]
        tol_values = [0.001, 0.01, 0.1]

        # Initialize outer and inner cross-validation splitters
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # List of datasets and names for easy iteration
        datasets = [
            #('sr1', X_tr100, y_tr100, group_tr100),
            #('sr2', X_tr2150, y_tr2150, group_tr2150),
            #('add', X_add, y_add, group_add),
            #('concat', X_concat, y_concat, group_concat),
            ('combcov', X_combcov, y_combcov, group_combcov)
        ]

        for name, X, y, group in datasets:
            logging.info(f"Starting nested cross-validation for {name}")

            best_auc = -float('inf')
            best_model = None
            best_params = None

            outer_cv_results = []

            # Outer loop
            for outer_fold_number, (outer_train_idx, outer_test_idx) in enumerate(sgkf.split(X, y, group)):
                X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
                y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
                group_outer_train, group_outer_test = group[outer_train_idx], group[outer_test_idx]

                logging.info(f"Outer Fold {outer_fold_number}: Training on subjects {set(group_outer_train)}, Testing on subjects {set(group_outer_test)}")

                # Initialize tracking variables for the inner loop
                best_inner_auc = -float('inf')
                best_inner_model = None
                best_inner_params = None

                # Inner loop for hyperparameter tuning
                for C in C_values:
                    for coef0 in coef0_values:
                        for tol in tol_values:
                            inner_cv_results = []

                            # Perform inner cross-validation
                            for inner_fold_number, (inner_train_idx, inner_val_idx) in enumerate(sgkf.split(X_outer_train, y_outer_train, group_outer_train)):
                                X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
                                y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
                                group_inner_train, group_inner_test = group[inner_train_idx], group[inner_val_idx]

                                # Initialize and train the ThunderSVM model
                                svm = SVC(kernel=kernel_type, C=C, coef0=coef0, tol=tol)
                                svm.fit(X_inner_train, y_inner_train)

                                # Predict and evaluate
                                y_pred = svm.predict(X_inner_val)
                                # Calculate AUC assuming binary classification
                                inner_auc = roc_auc_score(y_inner_val, y_pred)
                                inner_cv_results.append(inner_auc)

                                logging.info(f"Inner Fold {inner_fold_number}: C={C}, coef0={coef0}, tol={tol}, ROC AUC={inner_auc:.4f} subjects in train {set(group_inner_train)} subjects in test {set(group_inner_test)}")

                            # Average AUC for the current hyperparameter setting
                            mean_inner_auc = np.mean(inner_cv_results)

                            # Track the best model in the inner loop
                            if mean_inner_auc > best_inner_auc:
                                best_inner_auc = mean_inner_auc
                                best_inner_model = svm
                                best_inner_params = {'C': C, 'coef0': coef0, 'tol': tol}

                # Evaluate on the outer test set with the best model from inner loop
                if best_inner_model is not None:
                    y_pred = best_inner_model.predict(X_outer_test)
                    # Calculate AUC
                    outer_auc = roc_auc_score(y_outer_test, y_pred)
                    outer_cv_results.append(outer_auc)

                    logging.info(f"Outer Fold {outer_fold_number}: Best Parameters: C={best_inner_params['C']}, coef0={best_inner_params['coef0']}, tol={best_inner_params['tol']}, ROC AUC={outer_auc:.4f}")

                    # Track the best model across outer folds
                    if outer_auc > best_auc:
                        best_auc = outer_auc
                        best_model = best_inner_model
                        best_params = best_inner_params

            # Report results
            logging.info(f"Best model for {name}: C={best_params['C']}, coef0={best_params['coef0']}, tol={best_params['tol']}, ROC AUC={best_auc:.4f}")
            logging.info(f"Mean AUC across outer folds: {np.mean(outer_cv_results):.4f}")


            # Save the best model
            if best_model is not None:

                result_path = f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type}'
                filename = f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}.pkl'
                result_file = f'{result_path}/{filename}'

                with open(result_file, 'wb') as f:
                    pickle.dump(best_model, f)

        


if __name__ == "__main__":
    main()

