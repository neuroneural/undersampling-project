import sys
sys.path.append("/data/users2/jwardell1/undersampling-project/scripts/")

from utils.usp_utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np

from joblib import load

from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

from utils.polyssifier import *
from utils.usp_utils import *


from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

SNRs = [2.3]#[1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5]

def get_dataset(SNR, signal_dataset='oulu'):
    data_params = {}
    data_params['SNR'] = SNR
    signal_dataset = signal_dataset.upper()
    noise_dataset = 'fbirn'.upper()
    sampling_rates = ['sr1', 'sr2', 'add', 'concat']
    
    project_dir ='/data/users2/jwardell1/undersampling-project'
    signal_data = pd.read_pickle(f'{project_dir}/assets/data/{signal_dataset}_data.pkl')
    noise_data = pd.read_pickle(f'{project_dir}/assets/data/{noise_dataset}_data.pkl')
    n_folds = 7
    subject_id = '000300655084'
    L, correlation_matrix = get_subject_data(subject_id, noise_data)
    log_level = 'DEBUG'
    num_noise = 1
    cov_mat = False
    window_pairs = True

    undersampling_rate = 1 if signal_dataset == 'OULU' else 7
    NOISE_SIZE = 2961*2 if signal_dataset == 'OULU' else 1200
    subjects = np.unique(signal_data['subject'])


    

    data_params['project_dir'] = project_dir
    data_params['subjects'] = subjects
    data_params['noise_dataset'] = noise_dataset
    data_params['signal_dataset'] = signal_dataset
    data_params['SNRs'] = SNRs
    data_params['n_folds'] = n_folds
    data_params['log_level'] = log_level
    data_params['signal_dataset'] = signal_dataset
    data_params['noise_dataset'] = noise_dataset
    data_params['signal_data'] = signal_data
    data_params['noise_data'] = noise_data
    data_params['undersampling_rate'] = undersampling_rate
    data_params['NOISE_SIZE'] = NOISE_SIZE
    data_params["num_noise"] = num_noise
    data_params['cov_mat'] = cov_mat
    data_params['subject_id'] = subject_id 
    data_params['window_pairs'] = window_pairs
    data_params['correlation_matrix'] = correlation_matrix
    data_params['L'] = L


    




    signal_data = data_params['signal_data']
    n_folds = int(data_params['n_folds'])
    log_level = data_params['log_level']
    num_noise = data_params['num_noise']
    cov_mat = data_params['cov_mat']
    subject_id = data_params['subject_id']
    window_pairs = data_params['window_pairs']

    all_data = load_timecourses(signal_data, data_params)

    data_df = pd.DataFrame(all_data)



    ################ windowing
    sr1_data, sr2_data, add_data, concat_data = perform_windowing(data_df)

    sr1_df = pd.DataFrame(sr1_data)
    sr2_df = pd.DataFrame(sr2_data)

    X_sr1, y_sr1, group_sr1 = parse_X_y_groups(sr1_df, 'SR1')
    X_sr2, y_sr2, group_sr2 = parse_X_y_groups(sr2_df, 'SR2')



    # generates all combinations of window pairs, ordered by class label and subject id
    window_pairs, class_labels, group_labels = create_window_pairs(sr1_df, sr2_df)

    # shuffles the window pairs, preserves the class labels, and group labels
    windows_sh, class_sh, group_sh = shuffle_windows(window_pairs, class_labels, group_labels)

    # takes the first n window pairs, ordered by class label and subject id
    windows_st, class_st, group_st = take_first_n_windows(windows_sh, class_sh, group_sh)



    #use the random window combinations to generate the add and concat features
    X_add, y_add, group_add = get_combined_features(windows_st, class_st, group_st, type='add')
    X_concat, y_concat, group_concat = get_combined_features(windows_st, class_st, group_st, type='concat')

    # Apply label encoding
    label_encoder = LabelEncoder()
    y_sr1 = label_encoder.fit_transform(y_sr1)
    y_sr2 = label_encoder.transform(y_sr2)
    group_encoder = LabelEncoder()
    group_sr1 = group_encoder.fit_transform(group_sr1)
    group_sr2 = group_encoder.transform(group_sr2)
    
    # Standardize features
    s1 = StandardScaler()
    s2 = StandardScaler()
    s3 = StandardScaler()
    s4 = StandardScaler()
    
    X_sr1 = s1.fit_transform(X_sr1)
    X_sr2 = s2.fit_transform(X_sr2)
    
    # Generate window pairs and shuffled data
    window_pairs, class_labels, group_labels = create_window_pairs(sr1_df, sr2_df)
    windows_sh, class_sh, group_sh = shuffle_windows(window_pairs, class_labels, group_labels)
    windows_st, class_st, group_st = take_first_n_windows(windows_sh, class_sh, group_sh)
    
    # Generate combined features
    X_add, y_add, group_add = get_combined_features(windows_st, class_st, group_st, type='add')
    X_concat, y_concat, group_concat = get_combined_features(windows_st, class_st, group_st, type='concat')
    
    # Apply label encoding to additional datasets
    y_add = label_encoder.transform(y_add)
    y_concat = label_encoder.transform(y_concat)
    group_add = group_encoder.transform(group_add)
    group_concat = group_encoder.transform(group_concat)
    
    # Standardize additional datasets
    X_add = s3.fit_transform(X_add)
    X_concat = s4.fit_transform(X_concat)

    datasets = [
        ('sr1', X_sr1, y_sr1, group_sr1),
        ('sr2', X_sr2, y_sr2, group_sr2),
        ('add', X_add, y_add, group_add),
        ('concat', X_concat, y_concat, group_concat),
    ]
    return datasets





def load_results(signal_dataset = 'oulu', date = '02-06'):
    import os
    import glob
    import pandas as pd
    import numpy as np

    noise_dataset = 'fbirn'
    kernel_type = 'rbf'
    window_pairs = True

    if window_pairs:
        pkl_dir = f'/data/users2/jwardell1/undersampling-project/{signal_dataset.upper()}/pkl-files/{noise_dataset.upper()}/{date}/rwp'

    print(pkl_dir)

    # Function to load and process data
    def load_data(pattern, sampling_rate):
        joined_files = os.path.join(pkl_dir, pattern)
        joined_list = glob.glob(joined_files)
        data_frames = []
        for file in joined_list:
            try:
                df = pd.read_pickle(file)
                df['sampling_rate'] = sampling_rate
                df['subject_id'] = file.split('_')[5].split('.')[0]
                data_frames.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

    # Load data for all sampling rates
    sr1 = load_data('sr1_*.pkl', 'sr1')
    sr2 = load_data('sr2_*.pkl', 'sr2')
    concat = load_data('concat_*.pkl', 'concat')
    add = load_data('add_*.pkl', 'add')

    # Combine all data
    all_data = pd.concat([sr1, sr2, concat, add], ignore_index=True)
    return all_data

def get_best_fold(snr, sr, all_data):
    test_scores = all_data[
        (all_data['sampling_rate'] == sr) &
        (all_data['snr'] == snr)
    ]['test_scores']
    if(test_scores.shape[0] > 1):
        test_scores = test_scores.iloc[1]
    fold = np.argmax(test_scores)
    score = np.max(test_scores)
    if(not isinstance(score, np.number)):
        test_scores = score[0]
        fold = np.argmax(test_scores)
        score = np.max(test_scores)
        print(f'snr {snr} \t sr {sr} \t best fold {fold} \t auc {score}')
    else:
        print(f'snr {snr} \t sr {sr} \t best fold {fold} \t auc {score}')
    return fold



def get_lr_hps(snr, sr, fold):
    model_dir = f'/data/users2/jwardell1/undersampling-project/scripts/experiments/poly_{sr}-{snr}/models'
    lr_filename = f'Logistic Regression_{fold}.p'
    model = load(f'{model_dir}/{lr_filename}')
    C = model.best_estimator_.get_params()['steps'][1][1].C
    return C
    





def permutation_test(X, y, n_permutations, alpha, random_seed, w, d, reg=1):
    # Original coefficients
    original_coefs = np.abs(w).ravel()  # Use absolute values for two-tailed test

    # Store null distribution for each feature
    null_distribution = np.zeros((d, n_permutations))

    # Permutation testing loop
    np.random.seed(random_seed)
    for i in tqdm(range(n_permutations)):
        # Shuffle labels to break feature-label relationship
        y_permuted = np.random.permutation(y)
        # Train model on permuted labels
        model = LogisticRegression(penalty='l2', C=reg, solver='lbfgs', max_iter=1000)
        model.fit(X, y_permuted)
        # Store absolute coefficients of permuted model
        null_distribution[:, i] = np.abs(model.coef_.ravel())
        

    # Calculate p-values with pseudocount      
    pseudocount = 1
    p_values = (pseudocount + np.sum(null_distribution >= original_coefs[:, np.newaxis], axis=1)) / (n_permutations + pseudocount)

    # Adjust for multiple testing using Benjamini-Hochberg FDR control
    reject, fdr_corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    return p_values, fdr_corrected_pvals, reject



all_data = load_results(date='02-12', signal_dataset='hcp')

res = []


for SNR in SNRs:
    datasets = get_dataset(SNR, signal_dataset='hcp')
    for sr, X, y, group in datasets:
        best_fold = get_best_fold(SNR, sr, all_data)
        best_fold = best_fold+1
        C = get_lr_hps(SNR, sr, best_fold)
        print(f'sr: {sr} snr: {SNR} best_fold: {best_fold} C: {C}')
        #train lr model based on best model per snr, sr
        model = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2', C=C)
        model.fit(X, y)
        coefficients = model.coef_
        w = coefficients.ravel()
        n_permutations = 1000
        alpha=0.05
        random_seed=42
        d = X.shape[1]

        reg = C#0.2
    
        print(f'trying reg {reg}')
        p_values, fdr_corrected_pvals, reject = permutation_test(X, y, n_permutations, alpha, random_seed, w, d, reg)


        # print significant features
        significant_features = np.where(reject)[0]
        print(f'p_values: {p_values}')
        print(f'significant raw p_values: {np.where(p_values < 0.05)}')
        print(f'num significant raw p_values: {len(np.where(p_values < 0.05)[0])} of {w.shape[0]}')
        print(f'pvals_adjusted: {fdr_corrected_pvals}')
        print(f'significant_features: {significant_features}')
        print(f'num significant_features: {len(significant_features)} of {w.shape[0]}')


        plt.clf()
        sns.histplot(p_values, bins=50, kde=True)
        plt.xlabel("Raw p-values")
        plt.ylabel("Count")
        plt.title(f"Distribution of Raw p-values Before FDR Correction reg {reg}")
        plt.savefig(f'rawpvals_{reg}.png')

        res.append({
            'sr' : sr,
            'snr' : SNR,
            'p_values' : p_values,
            'fdr_corrected_pvals' : fdr_corrected_pvals,
            'reject' : reject,
            'w' : w,
            'reg' : reg
        })

res_df = pd.DataFrame(res)
date = datetime.now().strftime('%Y-%m-%d') + '-' + str(int(time.time()))
filename = f'results-{date}.pkl'
res_df.to_pickle(filename)
