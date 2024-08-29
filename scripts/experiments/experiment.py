import logging
import argparse

import pandas as pd
import numpy as np

import scipy.io

from utils.polyssifier import poly
from utils.usp_utils import *



def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'

    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name (FBIRN, COBRE, VAR)', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name (OULU, HCP)', required=True)

    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-nn', '--num_noise', type=int, help='number of noise iterations', required=False)
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
    
    
    n_threads = 1
    
    noise_dataset = args.noise_dataset.upper()
    signal_dataset = args.signal_dataset.upper()
    n_folds = args.n_folds if args.n_folds != None else 7
    num_noise = args.num_noise if args.num_noise != None else 1
    log_level = 'DEBUG' if args.verbose else 'INFO'

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Print the parsed arguments to verify
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Noise Iterations: {num_noise}')
    

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

        data_params['SNR'] = SNR

        for noise_ix in range(num_noise):

            ################ loading and preprocessing
            all_data = load_timecourses(signal_data, data_params)


            data_df = pd.DataFrame(all_data)



            ################ windowing
            sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)
            

            X_tr100, y_tr100, group_tr100 = parse_X_y_groups(pd.DataFrame(sr1_data), 'SR1')
            X_tr2150, y_tr2150, group_tr2150 = parse_X_y_groups(pd.DataFrame(sr2_data), 'SR2')
            X_add, y_add, group_add = parse_X_y_groups(pd.DataFrame(add_data), 'Add')
            X_concat, y_concat, group_concat = parse_X_y_groups(pd.DataFrame(concat_data), 'Concat')


            datasets = [
                ('sr1', X_tr100, y_tr100, group_tr100),
                ('sr2', X_tr2150, y_tr2150, group_tr2150),
                ('add', X_add, y_add, group_add),
                ('concat', X_concat, y_concat, group_concat)
            ]




            for name, X, y, group in datasets:
                logging.info(f'run polyssifier for for {name}')
                report = poly(data=X, label=y, groups=group, n_folds=n_folds, scale=True, concurrency=n_threads, save=False, 
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM', 'SVM'], scoring='auc', 
                            project_name=name)
                
                for classifier in report.scores.columns.levels[0]:
                    if classifier == 'Voting':
                        continue
                    
                    scores = report.scores[classifier, 'test']

                    results[name].append(
                        {
                            'noise_no': noise_ix,
                            'snr': SNR,
                            'classifier': classifier,
                            'test_scores': scores, 
                            'target': report.target, 
                            'predictions': np.array(report.predictions[classifier]).astype(int),
                            'test_proba': report.test_proba[classifier]
                        }
                    )
                    
                    

                    logging.info(f'{name} - SNR {SNR} - noise iteration {noise_ix} - scores {scores}')

        pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files/{noise_dataset}' if project_dir != '.' else '.'
        logging.info(f'pkl_dir: {pkl_dir}')

        for key, data in results.items():
            df = pd.DataFrame(data)
            filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}.pkl'
            df.to_pickle(f'{pkl_dir}/{filename}')
            logging.info(f'saved results for {key} at {pkl_dir}/{filename}')

if __name__ == "__main__":
    main()

