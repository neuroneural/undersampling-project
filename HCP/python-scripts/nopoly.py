import statsmodels.api as sm
import pandas as pd
import numpy as np
import sys
import logging
import scipy.sparse as sp
import scipy.io
from scipy.stats import zscore
from scipy.signal import detrend
from scipy.sparse.linalg import eigs
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import graph2adj
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SNRs = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

undersampling_factor = 3
L = np.load('/data/users2/jwardell1/cholesky_decomposition.npy')
covariance_matrix = np.load('/data/users2/jwardell1/covariance_matrix.npy')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


np.random.seed(42)
NOISE_SIZE = 1200
subjects = np.loadtxt("/data/users2/jwardell1/undersampling-project/HCP/txt-files/subjects.txt", dtype=str)
NUM_SUBS = len(subjects)

num_graphs = 3
num_noise = 5
n_folds = 2
n_threads = 5

logging.info(f'\t\t\t\tNUM_SUBS {NUM_SUBS}')


for SNR in SNRs:
    for noise_ix in range(num_noise):
        num_converged = 0
        converged_subjects = []
        noises = dict()
        while num_converged < NUM_SUBS:
            for subject in subjects:
                mean = np.zeros(covariance_matrix.shape[0])
                white_noise = np.random.multivariate_normal(mean, np.eye(covariance_matrix.shape[0]), size=NOISE_SIZE)
                colored_noise = white_noise @ L.T
                noises[subject] = colored_noise.T
                num_converged += 1
                converged_subjects.append(subject)
                logging.info(f'num converged: {num_converged}')
        
        all_data = []

        for i in range(NUM_SUBS):
            subject = subjects[i]
            logging.info(f'loading TC for subject {subject}')
            filepath_sr1 = f'/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/{subject}/processed/TCOutMax_{subject}.mat'
            try:
                sr1_tc = scipy.io.loadmat(filepath_sr1)['TCMax']
            except:
                continue

            if sr1_tc.shape[0] != 53:
                sr1_tc = sr1_tc.T

            if sr1_tc.shape[1] < 1200:
                continue

            logging.info(f'sr1.shape - {sr1_tc.shape}')
            sr1_tc_zc = zscore(sr1_tc, axis=1)
            sr1_tc_zc_dt = scipy.signal.detrend(sr1_tc_zc, axis=1)

            sr2_tc_zc_dt = sr1_tc_zc_dt[:, ::undersampling_factor]
            logging.info(f'sr2.shape - {sr2_tc_zc_dt.shape}')

            sr1_tc_zc_dt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(sr1_tc_zc_dt)
            sr2_tc_zc_dt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(sr2_tc_zc_dt)

            noise_sr1 = noises[subject]
            noise_sr2 = noises[subject][:, ::undersampling_factor]

            all_data.append({
                'Subject_ID': str(subject), 
                'VAR_Noise': noises[subject], 
                'SR1_Noise': noise_sr1, 
                'SR2_Noise': noise_sr2, 
                'SR1_Timecourse': sr1_tc_zc_dt, 
                'SR2_Timecourse': sr2_tc_zc_dt
            })
            
        data_df = pd.DataFrame(all_data)

        xTx_sr1 = np.sum(np.square(data_df['SR1_Timecourse'].mean()))
        nTn_sr1 = np.sum(np.square(data_df['SR1_Noise'].mean()))
        scalar_sr1 = ((xTx_sr1 / nTn_sr1)**0.5) / (10**(SNR / 2))

        xTx_sr2 = np.sum(np.square(data_df['SR2_Timecourse'].mean()))
        nTn_sr2 = np.sum(np.square(data_df['SR2_Noise'].mean()))
        scalar_sr2 = ((xTx_sr2 / nTn_sr2)**0.5) / (10**(SNR / 2))

        logging.info(f'\t\t\t\tSNR {SNR}')
        logging.info(f'\t\t\t\tscalar_sr1 {scalar_sr1}')
        logging.info(f'\t\t\t\tscalar_sr2 {scalar_sr2}')

        data_df['SR1_Noise'] = data_df['SR1_Noise'].multiply(scalar_sr1)
        data_df['SR2_Noise'] = data_df['SR2_Noise'].multiply(scalar_sr2)

        data_df['SR1_Timecourse_Noise'] = data_df['SR1_Noise'] + data_df['SR1_Timecourse']
        data_df['SR2_Timecourse_Noise'] = data_df['SR2_Noise'] + data_df['SR2_Timecourse']

        sr1_data = []
        sr2_data = []
        add_data = []
        concat_data = []
        groups = []
        targets = []

        for i, subject in enumerate(subjects):
            sub_row = data_df[data_df['Subject_ID'] == subject]
            logging.info(f'subject {subject}')

            sr1 = sub_row['SR1_Timecourse'].iloc[0]
            sr1_noise = sub_row['SR1_Timecourse_Noise'].iloc[0]
            
            sr2 = sub_row['SR2_Timecourse'].iloc[0]
            sr2_noise = sub_row['SR2_Timecourse_Noise'].iloc[0]

            n_regions, n_tp_sr1 = sr1.shape
            _, n_tp_sr2 = sr2.shape

            sr2_window_size = 100
            sr2_stride = 1
            n_sections = 80
            sr2_start_ix = 0
            sr2_end_ix = sr2_window_size
            
            sr1_window_size = int((n_tp_sr1 / n_tp_sr2) * sr2_window_size)
            sr1_stride = n_tp_sr1 // n_tp_sr2
            sr1_start_ix = 0
            sr1_end_ix = sr1_window_size

            for j in range(n_sections):
                sr1_section = sr1[:, sr1_start_ix:sr1_end_ix]
                sr1_section_noise = sr1_noise[:, sr1_start_ix:sr1_end_ix]
                
                sr1_start_ix += sr1_stride
                sr1_end_ix += sr1_stride

                sr2_section = sr2[:, sr2_start_ix:sr2_end_ix]
                sr2_section_noise = sr2_noise[:, sr2_start_ix:sr2_end_ix]

                sr2_start_ix += sr2_stride
                sr2_end_ix += sr2_stride

                sr1_data.append(sr1_section_noise)
                sr2_data.append(sr2_section_noise)
                add_data.append(sr1_section_noise + sr2_section_noise)
                concat_data.append(np.concatenate((sr1_section_noise, sr2_section_noise), axis=1))

                groups.append(subject)
                targets.append(0 if 'HC' in subject else 1)

        sr1_arr = np.array(sr1_data)
        sr2_arr = np.array(sr2_data)
        add_arr = np.array(add_data)
        concat_arr = np.array(concat_data)

        fold_roc_sr1 = []
        fold_roc_sr2 = []
        fold_roc_add = []
        fold_roc_concat = []

        sgkf = StratifiedGroupKFold(n_splits=n_folds)

        for train_index, test_index in sgkf.split(sr1_arr, targets, groups):
            sr1_train, sr1_test = sr1_arr[train_index], sr1_arr[test_index]
            sr2_train, sr2_test = sr2_arr[train_index], sr2_arr[test_index]
            add_train, add_test = add_arr[train_index], add_arr[test_index]
            concat_train, concat_test = concat_arr[train_index], concat_arr[test_index]

            y_train, y_test = np.array(targets)[train_index], np.array(targets)[test_index]

            svm_sr1 = SVC(probability=True)
            svm_sr2 = SVC(probability=True)
            svm_add = SVC(probability=True)
            svm_concat = SVC(probability=True)

            svm_sr1.fit(sr1_train.reshape(len(sr1_train), -1), y_train)
            svm_sr2.fit(sr2_train.reshape(len(sr2_train), -1), y_train)
            svm_add.fit(add_train.reshape(len(add_train), -1), y_train)
            svm_concat.fit(concat_train.reshape(len(concat_train), -1), y_train)

            fold_roc_sr1.append(roc_auc_score(y_test, svm_sr1.predict_proba(sr1_test.reshape(len(sr1_test), -1))[:, 1]))
            fold_roc_sr2.append(roc_auc_score(y_test, svm_sr2.predict_proba(sr2_test.reshape(len(sr2_test), -1))[:, 1]))
            fold_roc_add.append(roc_auc_score(y_test, svm_add.predict_proba(add_test.reshape(len(add_test), -1))[:, 1]))
            fold_roc_concat.append(roc_auc_score(y_test, svm_concat.predict_proba(concat_test.reshape(len(concat_test), -1))[:, 1]))

    avg_roc_sr1 = np.mean(fold_roc_sr1)
    avg_roc_sr2 = np.mean(fold_roc_sr2)
    avg_roc_add = np.mean(fold_roc_add)
    avg_roc_concat = np.mean(fold_roc_concat)

    logging.info(f'Average ROC AUC for sr1: {avg_roc_sr1}')
    logging.info(f'Average ROC AUC for sr2: {avg_roc_sr2}')
    logging.info(f'Average ROC AUC for add: {avg_roc_add}')
    logging.info(f'Average ROC AUC for concat: {avg_roc_concat}')
