import logging
import argparse
import pickle
from datetime import datetime
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import scipy.io

from thundersvm import SVC

from utils.usp_utils import *

def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'


    parser = argparse.ArgumentParser()

    
    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)

    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-nn', '--num_noise', type=int, help='number of noise iterations', required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='turn on debug logging', required=False)
    parser.add_argument('-o', '--optuna', action='store_true', help='use optuna weights', required=False)
    
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
    n_folds = args.n_folds if args.n_folds != None else 7
    kernel_type = args.kernel_type
    num_noise = args.num_noise if args.num_noise != None else 1
    log_level = 'DEBUG' if args.verbose else 'INFO'
    optuna = True if args.optuna else False

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Print the parsed arguments to verify
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Kernel Type: {kernel_type}')
    logging.info(f'Noise Iterations: {num_noise}')
    logging.info(f'Use Optuna Weights: {optuna}')


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
        res1 = []
        res2 = []
        res3 = []
        res4 = []


        for noise_ix in range(num_noise):
            data_params['SNR'] = SNR
            ################ loading and preprocessing
            all_data = load_timecourses(signal_data, data_params)



            data_df = pd.DataFrame(all_data)



            sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)
            

            X_tr100, y_tr100, group_tr100 = parse_X_y_groups(pd.DataFrame(sr1_data), 'SR1')
            X_tr2150, y_tr2150, group_tr2150 = parse_X_y_groups(pd.DataFrame(sr2_data), 'SR2')
            X_add, y_add, group_add = parse_X_y_groups(pd.DataFrame(add_data), 'Add')
            X_concat, y_concat, group_concat = parse_X_y_groups(pd.DataFrame(concat_data), 'Concat')


            scaler = StandardScaler()

            X_tr100 = scaler.fit_transform(X_tr100)
            X_tr2150 = scaler.fit_transform(X_tr2150)
            X_add = scaler.fit_transform(X_add)
            X_concat = scaler.fit_transform(X_concat)


            y_tr100 = np.where(y_tr100 == '0', -1, 1)
            y_tr2150 = np.where(y_tr2150 == '0', -1, 1)
            y_add = np.where(y_add == '0', -1, 1)
            y_concat = np.where(y_concat == '0', -1, 1)


            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=1)


            datasets = [
                ('sr1', X_tr100, y_tr100, group_tr100),
                ('sr2', X_tr2150, y_tr2150, group_tr2150),
                ('add', X_add, y_add, group_add),
                ('concat', X_concat, y_concat, group_concat)
            ]

            results = {
                'sr1': res1,
                'sr2': res2,
                'concat': res3,
                'add': res4
            }

            for name, X, y, group in datasets:
                _, ax = plt.subplots()
                #plot_cv_indices(sgkf, X, y, group, ax, n_folds, save_data, lw=10)
                for fold_ix, (train_index, test_index) in enumerate(sgkf.split(X, y, group), start=0):
                    fold_scores = []
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    logging.info(f'{name.upper()} - SNR {SNR} - noise_ix {noise_ix} - fold {fold_ix}')
                    logging.info(f'subjects in test {set(group[test_index])}')

                    # Load model weights and predict on test data
                    weights_dir = f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type.lower()}'
                    if optuna:
                        model_filename = f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}_optuna.pkl'
                    else:
                        model_filename = f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}.pkl'
                    
                    model_path = f'{weights_dir}/{model_filename}'

                    if optuna:
                        with open(model_path, 'rb') as file:
                            hp = pickle.load(file)
                            C = hp['C']
                            gamma = hp['gamma']
                            tol = hp['tol']
                            svm = SVC(kernel=kernel_type, C=C, gamma=gamma, tol=tol)
                            svm.fit(X_train, y_train)
                    else:
                        with open(model_path, 'rb') as file:
                            svm = pickle.load(file)

                    y_pred = svm.predict(X_test)

                    fold_score = roc_auc_score(y_test, np.array(y_pred))
                    fold_scores.append(fold_score)

                    #plot_and_save_confusion_matrix(y_test, y_pred, save_data)
                    
                    logging.info(f' SNR {SNR} - {name} - fold {fold_ix} - noise iteration {noise_ix} fold_auc {fold_score}')
                    

                    results[name].append(
                        {
                            'snr'              : SNR,
                            'fold'             : fold_ix, 
                            'roc'              : fold_score,
                            'sampling_rate'    : name,
                            'classifier'       : 'SVM'
                        }
                    )
                    
                avg_roc = np.mean(fold_scores)
                logging.info(f'Average ROC AUC for {name}: {avg_roc}')


        pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files/{noise_dataset}/SVM' if project_dir != '.' else '.'

        for key, data in results.items():
            if data != []:
                df = pd.DataFrame(data)
                current_date = datetime.now().strftime('%Y-%m-%d') + '-' + str(int(time.time()))
                month_date = '{}-{}'.format(datetime.now().strftime('%m'), datetime.now().strftime('%d'))

                if optuna:
                    filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}_SVM_{kernel_type}_{current_date}_optuna.pkl'
                else:
                    filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}_SVM_{kernel_type}_{current_date}.pkl'
                
                directory = Path(f'{pkl_dir}/{month_date}')
                directory.mkdir(parents=True, exist_ok=True)

                df.to_pickle(f'{pkl_dir}/{month_date}/{filename}')
                logging.info(f'saved results for {key} at {pkl_dir}/{month_date}/{filename}')

if __name__ == "__main__":
    main()

