import logging
import argparse
import pickle

import pandas as pd
import numpy as np

import scipy.io

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from utils.usp_utils import *
from thundersvm import SVC


def objective(trial: Trial, X, y, group, kernel_type, sgkf):
    # Hyperparameter search space
    C = trial.suggest_float('C', 1, 1e3, log=True)
    gamma = trial.suggest_float('gamma', 1e-5, 1, log=True)
    tol = trial.suggest_float('tol', 1e-6, 2, log=True)

    
    outer_cv_results = []

    # Perform cross-validation within the trial
    for outer_fold_number, (outer_train_idx, outer_test_idx) in enumerate(sgkf.split(X, y, group)):
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        # Initialize and train the ThunderSVM model
        svm = SVC(kernel=kernel_type, C=C, gamma=gamma, tol=tol)
        svm.fit(X_outer_train, y_outer_train)

        # Predict and evaluate
        y_pred = svm.predict(X_outer_test)
        outer_auc = roc_auc_score(y_outer_test, y_pred)
        outer_cv_results.append(outer_auc)

    # Return mean AUC across outer folds
    mean_auc = np.mean(outer_cv_results)
    return mean_auc


def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)


    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='turn on debug logging', required=False)
    parser.add_argument('-cv', '--cov-mat', action='store_true', help='use covariance matrix', required=False)

    args = parser.parse_args()

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
    n_folds = args.n_folds if args.n_folds != None else 7
    kernel_type = args.kernel_type
    log_level = 'DEBUG' if args.verbose else 'INFO'
    cov_mat = True if args.cov_mat else False
    
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')



    # Print the parsed arguments to verify
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Kernel Type: {kernel_type}')
    logging.info(f'Covariance matrix: {cov_mat}')
    logging.info(f'Correlation matrix: {not cov_mat}')



    data_params = {}
    data_params['noise_dataset'] = noise_dataset
    data_params['signal_dataset'] = signal_dataset

    signal_data = pd.read_pickle(f'{project_dir}/assets/data/{signal_dataset}_data.pkl')
    
    noise_data = scipy.io.loadmat(f'{project_dir}/assets/data/cov/{noise_dataset}_data.mat') if cov_mat \
        else scipy.io.loadmat(f'{project_dir}/assets/data/{noise_dataset}_data.mat')
    

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
        logging.debug(f'L {L}')
        data_params['L'] = L

        if cov_mat:
            covariance_matrix = noise_data['cov_mat']    
            logging.debug(f'covariance_matrix {covariance_matrix}')
            data_params['covariance_matrix'] = covariance_matrix
        else:
            correlation_matrix = noise_data['corr_mat']    
            logging.debug(f'correlation_matrix {correlation_matrix}')
            data_params['correlation_matrix'] = correlation_matrix


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

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for SNR in SNRs:
        data_params['SNR'] = SNR

        # Preprocess data
        all_data = load_timecourses(signal_data, data_params)
        data_df = pd.DataFrame(all_data)

        sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)
        datasets = [
            ('sr1', pd.DataFrame(sr1_data), 'SR1'),
            ('sr2', pd.DataFrame(sr2_data), 'SR2'),
            ('add', pd.DataFrame(add_data), 'Add'),
            ('concat', pd.DataFrame(concat_data), 'Concat')
        ]

        for name, dataset_df, window_type in datasets:
            X, y, group = parse_X_y_groups(dataset_df, window_type)
            X = StandardScaler().fit_transform(X)
            y = np.where(y == '0', -1, 1)

            # Run Optuna optimization
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(lambda trial: objective(trial, X, y, group, kernel_type, sgkf), n_trials=50)

            best_trial = study.best_trial

            logging.info(f"Best model for {name} (SNR {SNR}): {best_trial.params}, ROC AUC: {best_trial.value:.4f}")

            # Save best model hyperparameters and results
            result_path = f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type}'

            filename = f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}_optuna.pkl' if cov_mat \
                else f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}_optuna_corr.pkl'
            
            
            result_file = f'{result_path}/{filename}'

            with open(result_file, 'wb') as f:
                pickle.dump(best_trial.params, f)


if __name__ == "__main__":
    main()
