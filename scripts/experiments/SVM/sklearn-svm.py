import logging
import argparse
import pickle

import pandas as pd
import numpy as np

import scipy.io

from utils.usp_utils import *





def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'


    parser = argparse.ArgumentParser()

    
    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)

    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-nn', '--noise-iter', type=int, help='num noise iterations', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-hps', type=int, choices=[1, 2], help='run hyperparameter search', required=False)
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
    log_level = 'DEBUG' if args.verbose else 'INFO'
    noise_iter = args.noise_iter if args.noise_iter != None else 1
    hps = args.hps if args.hps != None else -1
    run_hps = (hps == 1) or (hps == 2)


    if args.n_folds != None:
        n_folds = args.n_folds

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Print the parsed arguments to verify
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Kernel Type: {kernel_type}')
    if args.n_folds : logging.info(f'Number of Folds: {n_folds}')
    if args.hps : logging.info(f'HPS: {hps}')
    

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
    else: 
        NOISE_SIZE = 1200
        undersampling_rate = 6

    

    data_params['NOISE_SIZE'] = NOISE_SIZE
    data_params['undersampling_rate'] = undersampling_rate
    
    k = 9

    all_snr_data = {} if hps == 2 else None


    for SNR in SNRs:
        res1 = []
        res2 = []
        res3 = []
        res4 = []

        results = {
            'sr1': res1,
            'sr2': res2,
            'concat': res3,
            'add': res4
        }

        for noise_ix in range(noise_iter):
            logging.info(f'compute data for SNR {SNR} noise_ix {noise_ix}')
            data_params['SNR'] = SNR

            all_data = load_timecourses(signal_data, data_params)

            data_df = pd.DataFrame(all_data)




            ################ windowing
            sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)
            

            X_tr100, y_tr100, group_tr100 = parse_X_y_groups(pd.DataFrame(sr1_data), 'SR1')
            X_tr2150, y_tr2150, group_tr2150 = parse_X_y_groups(pd.DataFrame(sr2_data), 'SR2')
            X_add, y_add, group_add = parse_X_y_groups(pd.DataFrame(add_data), 'Add')
            X_concat, y_concat, group_concat = parse_X_y_groups(pd.DataFrame(concat_data), 'Concat')


            

            # List of datasets and names for easy iteration
            datasets = [
                ('sr1', X_tr100, y_tr100, group_tr100),
                ('sr2', X_tr2150, y_tr2150, group_tr2150),
                ('add', X_add, y_add, group_add),
                ('concat', X_concat, y_concat, group_concat)
            ]

            if hps == 2: 
                all_snr_data[SNR] = datasets


            if run_hps:
                param_grid = {
                    'C'        : [0.1, 1.0, 10.0],
                    'tol'      : [0.001, 0.01, 0.1],
                    'k_values' : [5]#[3, 8, 9],
                }

                

            # Iterate over different datasets
            for name, X, y, group in datasets:
                if hps == 1:
                    param_grid['name'] = name
                    logging.info(f'{name.upper()}  - SNR {SNR} - begin hyper-parameter search ')

                    best_model = tune_svm(X, y, group, param_grid)
                    if best_model is not None:
                        result_path = f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type}'
                        filename = f'sklearn_{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}.pkl'
                        result_file = f'{result_path}/{filename}'

                        with open(result_file, 'wb') as f:
                            logging.info(f'saving best model at {f}')
                            pickle.dump(best_model, f)
                elif hps == 2:
                    continue
                else: 
                    logging.info(f'{name.upper()}  - SNR {SNR} - noise_ix {noise_ix} -  begin classification')
                    model_info = {
                        'weights_dir'     :  f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type.lower()}',
                        'model_filename'  :  f'sklearn_{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}.pkl',

                    } 
                    fold_scores = fit_svm(X, y, group, model_info, k=k)
                    for fold_ix, fold_score in fold_scores.items():
                        results[name].append(
                            {
                                'snr'              : SNR,
                                'fold'             : fold_ix, 
                                'roc'              : fold_score,
                                'sampling_rate'    : name,
                            }
                        )
                        logging.info(f'{name.upper()}  - SNR {SNR} - noise_ix {noise_ix} - fold score {fold_score}')

        if not run_hps:
            pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files/{noise_dataset}/SVM/sklearn' if project_dir != '.' else '.'

            for key, data in results.items():
                df = pd.DataFrame(data)
                filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}_SVM_{kernel_type}.pkl'
                df.to_pickle(f'{pkl_dir}/{filename}')
                logging.info(f'saved results for {key} at {pkl_dir}/{filename}')
    
    if hps == 2:
        logging.info(f'begin evaluation across weights')
        sampling_rates = ['sr1', 'sr2', 'add', 'concat']
        
        pickle_file_path = '/data/users2/jwardell1/undersampling-project/assets/data/skmodel_info.pkl'
        with open(pickle_file_path, 'rb') as file:
            model_info = pickle.load(file)
        

        best_model, best_auc = evaluate_weights_across_snr(all_snr_data, model_info, SNRs, sampling_rates, k)
        logging.info(f'best_auc {best_auc}')

        if best_model is not None:
            result_path = f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type}'
            filename = f'sklearn_{name}_best_model_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}_HPS2.pkl'
            result_file = f'{result_path}/{filename}'

        with open(result_file, 'wb') as f:
            logging.info(f'saving best model at {f}')
            pickle.dump(best_model, f)


                
                

if __name__ == "__main__":
    main()

