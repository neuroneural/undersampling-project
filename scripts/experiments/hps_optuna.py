import logging
import argparse
import pickle
from pathlib import Path
from datetime import * 

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.samplers import RandomSampler

from utils.usp_utils import *
from thundersvm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB



def objective(trial: Trial, X, y, group, model_type, kernel_type, sgkf, SNR, sampling_rate, results_dict):
    
    hps_params = {}


    # Hyperparameter search space
    if model_type == 'svm':
        C = trial.suggest_float('C', 1, 1e3, log=True)
        gamma = trial.suggest_float('gamma', 1e-5, 1, log=True)
        tol = trial.suggest_float('tol', 1e-6, 2, log=True)
        
        hps_params['C'] = C
        hps_params['gamma'] = gamma
        hps_params['tol'] = tol 

    if model_type == 'lr':
        C = trial.suggest_float('C', 0.001, 1, log=True)

        hps_params['C'] = C
    
    if model_type == 'mlp':
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 100)])
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)

        hps_params['hidden_layer_sizes'] = hidden_layer_sizes
        hps_params['activation'] = activation
        hps_params['solver'] = solver
        hps_params['alpha'] = alpha
        hps_params['learning_rate'] = learning_rate
        hps_params['learning_rate_init'] = learning_rate_init

    if model_type == 'nb':
            hps_params['nb_type'] = trial.suggest_categorical("nb_type", ["gaussian", "multinomial"])

            if hps_params['nb_type'] == "multinomial":
                hps_params['alpha'] = trial.suggest_float("alpha", 1e-3, 10, log=True)



    if model_type != 'nb':
        scaler = StandardScaler()
    if model_type == 'nb':
        if hps_params['nb_type'] == "multinomial":
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()

    X = scaler.fit_transform(X)
            
    if model_type == 'svm':
        y = np.where(y == '0', -1, 1)


    outer_fold_results = []


    # Perform cross-validation within the trial
    for outer_fold_number, (outer_train_idx, outer_test_idx) in enumerate(sgkf.split(X, y, group)):
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]



        # Initialize the model based on script argument
        if model_type == 'svm':
            model = SVC(kernel=kernel_type, C=hps_params['C'], gamma=hps_params['gamma'], tol=hps_params['tol'])
        
        if model_type == 'lr':
            model = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2', max_iter=150)

        if model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=hps_params['hidden_layer_sizes'],
                activation=hps_params['activation'],
                solver=hps_params['solver'],
                alpha=alpha,
                learning_rate=hps_params['learning_rate'],
                learning_rate_init=hps_params['learning_rate_init'],
                max_iter=200,
                random_state=42
            )
        
        if model_type == 'nb':
            if hps_params['nb_type'] == "gaussian":
                model = GaussianNB()

            if hps_params['nb_type'] == "multinomial":
                model = MultinomialNB(alpha=hps_params['alpha'])
            
        
        # Fit the model
        model.fit(X_outer_train, y_outer_train)


        # Predict and evaluate
        y_pred = model.predict(X_outer_test)


        outer_auc = roc_auc_score(y_outer_test, y_pred)

        results_dict[sampling_rate].append({
            'snr': SNR,
            'fold': outer_fold_number,
            'roc': outer_auc,
            'sampling_rate': sampling_rate,
            'classifier': model_type
        })


    mean_auc = np.mean([result['roc'] for result in outer_fold_results])

    # Return mean AUC across outer folds
    mean_auc = np.mean([result['roc'] for result in results_dict[sampling_rate]])
    return mean_auc

    


def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)


    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-v', '--verbose', type=bool, help='turn on debug logging', required=False)
    parser.add_argument('-g', '--sampler', type=str, choices=['tpe', 'random'], help='sampler type', required=False)

    args = parser.parse_args()
    

    data_params = set_data_params(args, project_dir)


    signal_dataset = data_params['signal_dataset']
    noise_dataset = data_params['noise_dataset']
    SNRs = data_params['SNRs']
    signal_data = data_params['signal_data']
    n_folds = int(data_params['n_folds'])
    sampler = data_params['sampler']
    if 'kernel_type' in data_params:
        kernel_type = data_params['kernel_type']
    else: 
        kernel_type = 'none'
    log_level = data_params['log_level']


    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Kernel Type: {kernel_type}')
    logging.info(f'Sampler Type: {sampler}')

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    #model_types = ['lr', 'mlp', 'svm', 'nb']
    model_types = ['svm']


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

        # Initialize results dictionaries for each sampling rate
        results_dict = {'sr1': [], 'sr2': [], 'add': [], 'concat': []}

        for sr, dataset_df, window_type in datasets:
            data_params['sr'] = sr
            data_params['SNR'] = SNR

            X, y, group = parse_X_y_groups(dataset_df, window_type)
            
            for model_type in model_types:            

                # Run Optuna optimization
                if sampler == 'tpe':
                    study = optuna.create_study(direction='maximize', sampler=TPESampler())
                else:
                    study = optuna.create_study(direction='maximize', sampler=RandomSampler())

                study.optimize(lambda trial: objective(trial, X, y, group, model_type, kernel_type, sgkf, SNR, sr, results_dict), n_trials=50)

                best_trial = study.best_trial

                logging.info(f"Best model for {sr} (SNR {SNR}): {best_trial.params}, ROC AUC: {best_trial.value:.4f}")

                # Save best model hyperparameters and results
                result_path = get_resultpath(data_params)
                
                filename = get_filename(data_params)
                    
                
                result_file = f'{result_path}/{filename}'

                directory = Path(result_path)
                directory.mkdir(parents=True, exist_ok=True)

                with open(result_file, 'wb') as f:
                    pickle.dump(best_trial.params, f)


            pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files/{noise_dataset}/optuna' if project_dir != '.' else '.'
            data_params['pkl_dir'] = pkl_dir


            # Save results to pickle files for each sampling rate and SNR
            for key, data in results_dict.items():
                write_results_to_pickle(data, data_params, key)

            
        


if __name__ == "__main__":
    main()