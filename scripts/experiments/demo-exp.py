import logging
import argparse
from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import numpy as np

from utils.polyssifier import polyr
from utils.usp_utils import *



def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'

    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-d', '--demographics-filepath', type=str, help='fullpath to demographics file', required=True)
    parser.add_argument('-t', '--demo_type', type=str, help='type of demographic to use', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name (OULU, HCP)', required=True)
    

    parser.add_argument('-u', '--us-rate', type=str, help='undersampling rate for HCP dataset', required=False)

    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='turn on debug logging', required=False)
    parser.add_argument('-w', '--window-pairs', action='store_true', help='use random combinations of windows', required=False)
    
    
    args = parser.parse_args()
    data_params = set_data_params(args, project_dir)


    signal_dataset = data_params['signal_dataset']
    demographics_data = data_params['demographics_data']
    demo_type = data_params['demo_type']
    signal_data = data_params['signal_data']
    n_folds = int(data_params['n_folds'])
    log_level = data_params['log_level']
    window_pairs = data_params['window_pairs']
    us_rate = data_params['undersampling_rate']

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')



    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Random Window Pairs: {window_pairs}')
    if signal_dataset.lower() == 'hcp':
        logging.info(f'Undersampling rate for HCP dataset: {us_rate}')

    

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



    ################ loading and preprocessing
    all_data = load_timecourses(signal_data, data_params)

    data_df = pd.DataFrame(all_data)


    ################ windowing
    sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)
    
    sr1_df = pd.DataFrame(sr1_data)
    sr2_df = pd.DataFrame(sr2_data)

    X_sr1, y_sr1, group_sr1 = parse_X_y_groups(sr1_df, 'SR1') #TODO modify these functions to parse labels from demographics file 
    X_sr2, y_sr2, group_sr2 = parse_X_y_groups(sr2_df, 'SR2') #TODO modify these functions to parse labels from demographics file 


    if window_pairs:
        # generates all combinations of window pairs, ordered by class label and subject id
        window_pairs, class_labels, group_labels = create_window_pairs(sr1_df, sr2_df)

        # shuffles the window pairs, preserves the class labels, and group labels
        windows_sh, class_sh, group_sh = shuffle_windows(window_pairs, class_labels, group_labels)

        # takes the first n window pairs, ordered by class label and subject id
        windows_st, class_st, group_st = take_first_n_windows(windows_sh, class_sh, group_sh)



        #use the random window combinations to generate the add and concat features
        X_add, y_add, group_add = get_combined_features(windows_st, class_st, group_st, type='add')
        X_concat, y_concat, group_concat = get_combined_features(windows_st, class_st, group_st, type='concat')
    
    else:
        X_add, y_add, group_add = parse_X_y_groups(pd.DataFrame(add_data), 'Add')
        X_concat, y_concat, group_concat = parse_X_y_groups(pd.DataFrame(concat_data), 'Concat')



    datasets = [
        ('sr1', X_sr1, y_sr1, group_sr1),
        ('sr2', X_sr2, y_sr2, group_sr2),
        ('add', X_add, y_add, group_add),
        ('concat', X_concat, y_concat, group_concat),
    ]



    for sr, X, y, group in datasets:
        data_params['name'] = sr


        report = polyr(data=X, label=y, groups=group, n_folds=n_folds, scale=True, concurrency=1, save=False, 
                    exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'], scoring='auc', 
                    project_name=sr)
        
        for classifier in report.scores.columns.levels[0]:
            if classifier == 'Voting':
                continue
            
            scores = report.scores[classifier, 'test']

            results[sr].append(
                {
                    'classifier': classifier,
                    'test_scores': scores, 
                    'target': report.target, 
                    'predictions': np.array(report.predictions[classifier]).astype(int),
                    'test_proba': report.test_proba[classifier],
                    'time' : str(int(time.time())),
                    'us_rate' : us_rate

                }
            )





    pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files/demo-exp' if project_dir != '.' else '.' #TODO create folder for demo exps
    logging.info(f'pkl_dir: {pkl_dir}')

    for key, data in results.items():
        if data != []:
            df = pd.DataFrame(data)
            
            current_date = datetime.now().strftime('%Y-%m-%d') + '-' + str(int(time.time()))
            month_date = '{}-{}'.format(datetime.now().strftime('%m'), datetime.now().strftime('%d'))


            filename = f'{key}_{signal_dataset}_{current_date}_usrate_{us_rate}.pkl' if not window_pairs \
                else f'{key}_{signal_dataset}_{current_date}_usrate_{us_rate}_rwp.pkl'
            
            directory = Path(f'{pkl_dir}/{month_date}') if not window_pairs else Path(f'{pkl_dir}/{month_date}/rwp')
            directory.mkdir(parents=True, exist_ok=True)

            df.to_pickle(f'{directory}/{filename}')
            logging.info(f'Saved results for {key} at {directory}/{filename}')

if __name__ == "__main__":
    main()
