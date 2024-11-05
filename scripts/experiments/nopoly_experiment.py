import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from utils.usp_utils import *
from thundersvm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)


    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-nn', '--num_noise', type=int, help='number of noise iterations', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-v', '--verbose', type=bool, help='turn on debug logging', required=False)

    args = parser.parse_args()
    

    data_params = set_data_params(args, project_dir)


    signal_dataset = data_params['signal_dataset']
    noise_dataset = data_params['noise_dataset']
    kernel_type = data_params['kernel_type']
    SNRs = data_params['SNRs']
    num_noise = data_params['num_noise']
    signal_data = data_params['signal_data']
    n_folds = int(data_params['n_folds'])
    log_level = data_params['log_level']
    sampler = data_params['sampler']
    
    print(log_level)


    logger = logging.getLogger('my_custom_logger')
    logger.setLevel(log_level)

    # Create console handler and set level and format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Remove the default handlers if they exist to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        logger.addHandler(console_handler)
    


    
    logger.info(f'Noise Interval: {SNRs}')
    logger.info(f'Noise Dataset: {noise_dataset}')
    logger.info(f'Signal Dataset: {signal_dataset}')
    logger.info(f'Number of Folds: {n_folds}')
    logger.info(f'Kernel Type: {kernel_type}')


 
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

            classifiers = [
                ('Logistic Regression' , 'lr'),
                #('Multilayer Perceptron' , 'mlp'),
                ('SVM' , 'svm'),
                ('Naive Bayes' , 'nb')
            ]

            
            for name, X, y, group in datasets:     
                for classifier, clf_shortname in classifiers:



                    # Load model weights and predict on test data

                    logger.info(f'Load model weights for {SNR} - {name} - {clf_shortname}')
                    weights_dir = f'{project_dir}/assets/model_weights/{signal_dataset}/{kernel_type.lower()}' if clf_shortname == 'svm' \
                        else f'{project_dir}/assets/model_weights/{signal_dataset}/{clf_shortname.lower()}'
                    
                    model_filename = f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}_optuna.pkl' if clf_shortname == 'svm' \
                        else f'{name}_best_model_SNR_{SNR}_{clf_shortname.upper()}_{signal_dataset}_{noise_dataset}_optuna_{sampler}.pkl'
                    
                    model_path = f'{weights_dir}/{model_filename}'

                    
                    with open(model_path, 'rb') as file:
                        hp = pickle.load(file)

                    if clf_shortname == 'lr':
                        C = hp['C']
                        model = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2', C=C)

                    if clf_shortname == 'mlp':
                        hidden_layer_sizes = hp['hidden_layer_sizes']
                        activation = hp['activation']
                        solver = hp['solver']
                        alpha = hp['alpha']
                        learning_rate = hp['learning_rate']
                        learning_rate_init = hp['learning_rate_init']
                        model = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            solver=solver,
                            alpha=alpha,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init,
                            max_iter=1000,
                            random_state=42
                        )

                    if clf_shortname == 'svm':
                        C = hp['C']
                        gamma = hp['gamma']
                        tol = hp['tol']
                        model = SVC(kernel=kernel_type, C=C, gamma=gamma, tol=tol)
                    
                    if clf_shortname == 'nb':
                        nb_type = hp['nb_type']
                        if nb_type == 'gaussian':
                            model = GaussianNB()
                        else:
                            model = MultinomialNB(alpha=hp['alpha'])




                    if clf_shortname == 'nb':
                        if nb_type != 'gaussian':
                            scaler = MinMaxScaler(feature_range=(0,1))
                        else:
                            scaler = StandardScaler()
                    else:
                        scaler = StandardScaler()


                    X = scaler.fit_transform(X)

                    if clf_shortname == 'svm':
                        y = np.where(y == '0', -1, 1)


                    for fold_ix, (train_index, test_index) in enumerate(sgkf.split(X, y, group), start=0):
                        fold_scores = []
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        logger.info(f'{name.upper()} - SNR {SNR} - noise_ix {noise_ix} - fold {fold_ix}')
                        logger.info(f'subjects in test {set(group[test_index])}')

                        
                        logger.info(f'X.shape {X.shape}')
                        logger.info(f'y.shape {y.shape}')
                        model.fit(X_train, y_train)

                        y_pred = model.predict(X_test)

                        fold_score = roc_auc_score(y_test, np.array(y_pred))
                        fold_scores.append(fold_score)

                        #plot_and_save_confusion_matrix(y_test, y_pred, save_data)
                        
                        logger.info(f' SNR {SNR} - {name} - fold {fold_ix} - noise iteration {noise_ix} fold_auc {fold_score}')
                        

                        results[name].append(
                            {
                                'snr'              : SNR,
                                'fold'             : fold_ix, 
                                'roc'              : fold_score,
                                'sampling_rate'    : name,
                                'classifier'       : classifier
                            }
                        )
                        
                    avg_roc = np.mean(fold_scores)
                    logger.info(f'Average ROC AUC for {name}: {avg_roc}')




        #TODO: FIX THE SAVING OF RESULTS
        pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files/{noise_dataset}/optuna' if project_dir != '.' else '.'
        directory = Path(f'{pkl_dir}')
        directory.mkdir(parents=True, exist_ok=True)

        for key, data in results.items():
            if data != []:
                df = pd.DataFrame(data)
                current_date = datetime.now().strftime('%Y-%m-%d') + '-' + str(int(time.time()))
                month_date = '{}-{}'.format(datetime.now().strftime('%m'), datetime.now().strftime('%d'))

                
                filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}{clf_shortname}{kernel_type}_{current_date}.pkl'
                
                directory = Path(f'{pkl_dir}/{month_date}')
                directory.mkdir(parents=True, exist_ok=True)

                df.to_pickle(f'{pkl_dir}/{month_date}/{filename}')
                logger.info(f'saved results for {key} at {pkl_dir}/{month_date}/{filename}')

if __name__ == "__main__":
    main()

