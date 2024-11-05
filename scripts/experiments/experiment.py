import logging
import argparse
from datetime import datetime

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
    parser.add_argument('-cv', '--cov-mat', action='store_true', help='use covariance matrix', required=False)
    
    args = parser.parse_args()
    data_params = set_data_params(args, project_dir)


    signal_dataset = data_params['signal_dataset']
    noise_dataset = data_params['noise_dataset']
    SNRs = data_params['SNRs']
    signal_data = data_params['signal_data']
    n_folds = int(data_params['n_folds'])
    log_level = data_params['log_level']
    num_noise = data_params['num_noise']
    cov_mat = data_params['cov_mat']

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Noise Iterations: {num_noise}')
    logging.info(f'Use correlation matrix: {not cov_mat}')
    logging.info(f'Use covariance matrix: {cov_mat}')

    

    for SNR in SNRs:
        res1 = []
        res2 = []
        res3 = []
        res4 = []

        results = {
            'sr1': res1,
            'sr2': res2,
            'concat': res3,
            'add': res4,
        }

        data_params['SNR'] = SNR

        for noise_ix in range(num_noise):

            ################ loading and preprocessing
            all_data = load_timecourses(signal_data, data_params)

            data_df = pd.DataFrame(all_data)


            ################ windowing
            sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)
            

            X_sr1, y_sr1, group_sr1 = parse_X_y_groups(pd.DataFrame(sr1_data), 'SR1')
            X_sr2, y_sr2, group_sr2 = parse_X_y_groups(pd.DataFrame(sr2_data), 'SR2')
            X_add, y_add, group_add = parse_X_y_groups(pd.DataFrame(add_data), 'Add')
            X_concat, y_concat, group_concat = parse_X_y_groups(pd.DataFrame(concat_data), 'Concat')



            datasets = [
                ('sr1', X_sr1, y_sr1, group_sr1),
                ('sr2', X_sr2, y_sr2, group_sr2),
                ('add', X_add, y_add, group_add),
                ('concat', X_concat, y_concat, group_concat),
            ]



            for name, X, y, group in datasets:
                logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(SNR)s - %(noise_ix)s - %(name)s - %(message)s')
                logging.info(f'run polyssifier for for {name}')
                report = poly(data=X, label=y, groups=group, n_folds=n_folds, scale=True, concurrency=1, save=False, 
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'], scoring='auc', 
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
            if data != []:
                df = pd.DataFrame(data)
                current_date = datetime.now().strftime('%Y-%m-%d')
                filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}_{current_date}.pkl'
                df.to_pickle(f'{pkl_dir}/{filename}')
                logging.info(f'saved results for {key} at {pkl_dir}/{filename}')

if __name__ == "__main__":
    main()