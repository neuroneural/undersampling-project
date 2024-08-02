import pandas as pd
import numpy as np
import logging
import scipy.io
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from libsvm.svmutil import *


SNRs = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

undersampling_factor = 6
L = np.load('/data/users2/jwardell1/nshor_docker/examples/fbirn-project/COV/000300655084_chol.npy')
covariance_matrix = np.load('/data/users2/jwardell1/nshor_docker/examples/fbirn-project/COV/000300655084_cov.npy')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


np.random.seed(42)
NOISE_SIZE = 1200
subjects = np.loadtxt("/data/users2/jwardell1/undersampling-project/HCP/txt-files/subjects.txt", dtype=str)
NUM_SUBS = len(subjects)


num_noise = 3
n_folds = 5


logging.info(f'\t\t\t\tNUM_SUBS {NUM_SUBS}')



for SNR in SNRs:
    res1 = []
    res2 = []
    res3 = []
    res4 = []
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

                sr2_section = sr2[:, sr2_start_ix:sr2_end_ix]
                sr2_section_noise = sr2_noise[:, sr2_start_ix:sr2_end_ix]






                sr1_fnc_triu = np.corrcoef(sr1_section)[np.triu_indices(n_regions)]
                sr1_noise_fnc_triu = np.corrcoef(sr1_section_noise)[np.triu_indices(n_regions)]   #TODO: debug


                sr2_fnc_triu = np.corrcoef(sr2_section)[np.triu_indices(n_regions)]
                sr2_noise_fnc_triu = np.corrcoef(sr2_section_noise)[np.triu_indices(n_regions)]


                concat_sr1_sr2 = np.concatenate((sr1_fnc_triu , sr2_fnc_triu))
                concat_sr1_sr2_noise = np.concatenate((sr1_noise_fnc_triu , sr2_noise_fnc_triu))

                add_sr1_sr2 = sr1_fnc_triu + sr2_fnc_triu
                add_sr1_sr2_noise = sr1_noise_fnc_triu + sr2_noise_fnc_triu


                sr1_data.append({'subject'          : subject, 
                                'SR1_Window'   : sr1_fnc_triu, 
                                'target'           : '0'})
                sr1_data.append({'subject'          : subject, 
                                'SR1_Window'   : sr1_noise_fnc_triu, 
                                'target'           : '1'})
                
                sr2_data.append({'subject'         : subject,
                                    'SR2_Window' : sr2_fnc_triu, 
                                    'target'          : '0'})
                sr2_data.append({'subject'         : subject,
                                    'SR2_Window' : sr2_noise_fnc_triu, 
                                    'target'          : '1'})
                
                concat_data.append({'subject'          : subject, 
                                    'Concat_Window'   : concat_sr1_sr2,
                                'target'            : '0'})
                concat_data.append({'subject'          : subject, 
                                    'Concat_Window'    : concat_sr1_sr2_noise,
                                    'target'           : '1'})
                
                add_data.append({'subject'             : subject,
                                'Add_Window'         : add_sr1_sr2,
                                'target'             : '0'})
                add_data.append({'subject'             : subject,
                                'Add_Window'         : add_sr1_sr2_noise,
                                'target'             : '1'})
                

                sr1_start_ix += sr1_stride
                sr1_end_ix = sr1_start_ix + sr1_window_size
                
                sr2_start_ix += sr2_stride
                sr2_end_ix = sr2_start_ix + sr2_window_size
                        

        sr1_df = pd.DataFrame(sr1_data)
        sr2_df = pd.DataFrame(sr2_data)
        concat_df = pd.DataFrame(concat_data)
        add_df = pd.DataFrame(add_data)

        fold_roc_sr1 = []
        fold_roc_sr2 = []
        fold_roc_add = []
        fold_roc_concat = []

        group_sr1 = sr1_df['subject']
        y_sr1 = sr1_df['target']
        y_sr1 = np.array([str(entry) for entry in y_sr1])
        X_sr1 = sr1_df['SR1_Window']
        X_sr1 = np.array([np.array(entry) for entry in X_sr1])
        
        
        group_sr2 = sr2_df['subject']
        y_sr2 = sr2_df['target']
        y_sr2 = np.array([str(entry) for entry in y_sr2])
        X_sr2 = sr2_df['SR2_Window']
        X_sr2 = np.array([np.array(entry) for entry in X_sr2])

        
        group_concat = concat_df['subject']
        y_concat = concat_df['target']
        y_concat = np.array([str(entry) for entry in y_concat])
        X_concat = concat_df['Concat_Window']
        X_concat = np.array([np.array(entry) for entry in X_concat])


        group_add = add_df['subject']
        y_add = add_df['target']
        y_add = np.array([str(entry) for entry in y_add])
        X_add = add_df['Add_Window']
        X_add = np.array([np.array(entry) for entry in X_add])



        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=1)

        logging.info(f'                  START LIBSVM FOR SNR {SNR} NOISE_IX {noise_ix}')

        for fold_number, (train_index, test_index) in enumerate(sgkf.split(X_sr1, y_sr1, group_sr1), start=0):
            sr1_train, sr1_test = X_sr1[train_index], X_sr1[test_index]
            sr2_train, sr2_test = X_sr2[train_index], X_sr2[test_index]
            add_train, add_test = X_add[train_index], X_add[test_index]
            concat_train, concat_test = X_concat[train_index], X_concat[test_index]

            y_train_sr1, y_test_sr1 = np.array(y_sr1)[train_index], np.array(y_sr1)[test_index]
            y_train_sr2, y_test_sr2 = np.array(y_sr2)[train_index], np.array(y_sr2)[test_index]
            y_train_add, y_test_add = np.array(y_add)[train_index], np.array(y_add)[test_index]
            y_train_concat, y_test_concat = np.array(y_concat)[train_index], np.array(y_concat)[test_index]

            # Prepare data for libsvm
            sr1_train_flat = sr1_train.reshape(len(sr1_train), -1).tolist()
            sr1_test_flat = sr1_test.reshape(len(sr1_test), -1).tolist()

            sr2_train_flat = sr2_train.reshape(len(sr2_train), -1).tolist()
            sr2_test_flat = sr2_test.reshape(len(sr2_test), -1).tolist()

            add_train_flat = add_train.reshape(len(add_train), -1).tolist()
            add_test_flat = add_test.reshape(len(add_test), -1).tolist()

            concat_train_flat = concat_train.reshape(len(concat_train), -1).tolist()
            concat_test_flat = concat_test.reshape(len(concat_test), -1).tolist()

            # Train and predict using libsvm
            svm_sr1 = svm_train(y_train_sr1.astype(int).tolist(), sr1_train_flat, '-b 1')
            svm_sr2 = svm_train(y_train_sr2.astype(int).tolist(), sr2_train_flat, '-b 1')
            svm_add = svm_train(y_train_add.astype(int).tolist(), add_train_flat, '-b 1')
            svm_concat = svm_train(y_train_concat.astype(int).tolist(), concat_train_flat, '-b 1')

            _, p_acc_sr1, p_vals_sr1 = svm_predict(y_test_sr1.astype(int).tolist(), sr1_test_flat, svm_sr1, '-b 1')
            _, p_acc_sr2, p_vals_sr2 = svm_predict(y_test_sr2.astype(int).tolist(), sr2_test_flat, svm_sr2, '-b 1')
            _, p_acc_add, p_vals_add = svm_predict(y_test_add.astype(int).tolist(), add_test_flat, svm_add, '-b 1')
            _, p_acc_concat, p_vals_concat = svm_predict(y_test_concat.astype(int).tolist(), concat_test_flat, svm_concat, '-b 1')

            fold_roc_sr1.append(roc_auc_score(y_test_sr1, np.array(p_vals_sr1)[:, 1]))
            fold_roc_sr2.append(roc_auc_score(y_test_sr2, np.array(p_vals_sr2)[:, 1]))
            fold_roc_add.append(roc_auc_score(y_test_add, np.array(p_vals_add)[:, 1]))
            fold_roc_concat.append(roc_auc_score(y_test_concat, np.array(p_vals_concat)[:, 1]))

            res1.append({'snr'                 : SNR,
                            'fold'             : fold_number, 
                            'roc'              : fold_roc_sr1[fold_number],
                            'sampling_rate'   : 'sr1',
                            'noise_ix'         : noise_ix
                        })
            
            res2.append({'snr'                 : SNR,
                            'fold'             : fold_number, 
                            'roc'              : fold_roc_sr2[fold_number],
                            'sampling_rate'   : 'sr2',
                            'noise_ix'         : noise_ix
                        })
            
            res3.append({'snr'                 : SNR,
                            'fold'             : fold_number, 
                            'roc'              : fold_roc_add[fold_number],
                            'sampling_rate'    : 'Add',
                            'noise_ix'         : noise_ix
                        })
            
            res4.append({'snr'                 : SNR,
                            'fold'             : fold_number, 
                            'roc'              : fold_roc_concat[fold_number],
                            'sampling_rate'   : 'Concat', 
                            'noise_ix'         : noise_ix
                        })

        avg_roc_sr1 = np.mean(fold_roc_sr1)
        avg_roc_sr2 = np.mean(fold_roc_sr2)
        avg_roc_add = np.mean(fold_roc_add)
        avg_roc_concat = np.mean(fold_roc_concat)

        logging.info(f'Average ROC AUC for sr1: {avg_roc_sr1}')
        logging.info(f'Average ROC AUC for sr2: {avg_roc_sr2}')
        logging.info(f'Average ROC AUC for add: {avg_roc_add}')
        logging.info(f'Average ROC AUC for concat: {avg_roc_concat}')


    df1 = pd.DataFrame(res1)  
    df1.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/sr1_{SNR}_libsvm.pkl')
    df2 = pd.DataFrame(res2)
    df2.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/sr2_{SNR}_libsvm.pkl')
    df3 = pd.DataFrame(res3)
    df3.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/concat_{SNR}_libsvm.pkl')
    df4 = pd.DataFrame(res4)
    df4.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/add_{SNR}_libsvm.pkl')






