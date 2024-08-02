import pandas as pd
import numpy as np
import logging
import scipy.io
from scipy.signal import detrend
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SNRs = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

undersampling_factor = 1
L = np.load('/data/users2/jwardell1/cholesky_decomposition.npy')
covariance_matrix = np.load('/data/users2/jwardell1/covariance_matrix.npy')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


np.random.seed(42)
NOISE_SIZE = 2961*2
NUM_SUBS = 10
subjects = ['20150210', '20150417', '20150428', '20151110', '20151127', 
            '20150410', '20150421', '20151030', '20151117', '20151204']


num_graphs = 3
num_noise = 5
n_folds = 2
n_threads = 5

logging.info(f'\t\t\t\tNUM_SUBS {NUM_SUBS}')



for SNR in SNRs:
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

    with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/allsubs_TCs.txt', 'r') as tc_data:
        lines = tc_data.readlines()

    for i in range(0, len(lines), 2):
        subject = subjects[i//2]
        logging.info(f'loading TC for subject {subject}')
        filepath_sr1 = lines[i].strip()
        filepath_sr2 = lines[i+1].strip()
        try:
            sr1 = scipy.io.loadmat(filepath_sr1)['TCMax']
            sr2 = scipy.io.loadmat(filepath_sr2)['TCMax']
        
        except:
            continue

        if sr1.shape[0] != 53:
            sr1 = sr1.T

        if sr2.shape[0] != 53:
            sr2 = sr2.T
        
        if sr1.shape[1] < sr2.shape[1]:
            tr100_tc = sr2
            tr2150_tc = sr1
        else:
            tr100_tc = sr1
            tr2150_tc = sr2

        logging.info(f'subject {subject} tr100_tc.shape - {tr100_tc.shape}')
        logging.info(f'subject {subject} tr2150_tc.shape - {tr2150_tc.shape}')

        tr100_tc_zs = zscore(tr100_tc, axis=1)
        tr2150_tc_zs = zscore(tr2150_tc, axis=1)

        tr100_tc_zs_dt = detrend(tr100_tc_zs, axis=1)
        tr2150_tc_zs_dt = detrend(tr2150_tc_zs, axis=1)

        tr100_tc_zs_dt = MinMaxScaler(feature_range=(-1,1)).fit_transform(tr100_tc_zs_dt)             #TRY MINMAX SCALING might remove
        tr2150_tc_zs_dt = MinMaxScaler(feature_range=(-1,1)).fit_transform(tr2150_tc_zs_dt)           #TRY MINMAX SCALING might remove

        noise_tr100 = noises[subject][:,::2]
        noise_tr2150 = noises[subject][:,::33]

        tr100_tc_zs_dt_noise = tr100_tc_zs_dt+noise_tr100
        tr2150_tc_zs_dt_noise = tr2150_tc_zs_dt+noise_tr2150

        all_data.append({'Subject_ID'             : str(subject), 
                        'VAR_Noise'               : noises[subject], 
                        'TR100_Noise'             : noise_tr100, 
                        'TR2150_Noise'            : noise_tr2150, 
                        'TR100_Timecourse'        : tr100_tc_zs_dt, 
                        'TR2150_Timecourse'       : tr2150_tc_zs_dt
                        })
            
    data_df = pd.DataFrame(all_data)
    
    xTx_tr100 = np.sum(np.square(data_df['TR100_Timecourse'].mean()))
    nTn_tr100 = np.sum(np.square(data_df['TR100_Noise'].mean()))
    scalar_tr100 = ((xTx_tr100 / nTn_tr100)**0.5) / (10**(SNR/2)) 

    xTx_tr2150 = np.sum(np.square(data_df['TR2150_Timecourse'].mean()))
    nTn_tr2150 = np.sum(np.square(data_df['TR2150_Noise'].mean()))
    scalar_tr2150 = ((xTx_tr2150 / nTn_tr2150)**0.5) / (10**(SNR/2)) 

    logging.info(f'\t\t\t\tSNR {SNR}')
    logging.info(f'\t\t\t\tscalar_tr100 {scalar_tr100}')
    logging.info(f'\t\t\t\tscalar1_tr2150 {scalar_tr2150}')

    data_df['TR100_Noise'] = data_df['TR100_Noise'].multiply(scalar_tr100)
    data_df['TR2150_Noise'] = data_df['TR2150_Noise'].multiply(scalar_tr2150)

    data_df['TR100_Timecourse_Noise'] = data_df['TR100_Noise'] + data_df['TR100_Timecourse']
    data_df['TR2150_Timecourse_Noise'] = data_df['TR2150_Noise'] + data_df['TR2150_Timecourse']

    tr100_data = []
    tr2150_data = []
    add_data = []
    concat_data = []
    for subject in subjects:
        sub_row = data_df[data_df['Subject_ID']  == subject]
        logging.info(f'subject {subject}')

        tr100 = sub_row['TR100_Timecourse'].iloc[0]
        tr100_noise = sub_row['TR100_Timecourse_Noise'].iloc[0]
        
        tr2150 = sub_row['TR2150_Timecourse'].iloc[0]
        tr2150_noise = sub_row['TR2150_Timecourse_Noise'].iloc[0]

        n_regions, n_tp_tr100 = tr100.shape
        _, n_tp_tr2150 = tr2150.shape

        tr2150_window_size = 100
        tr2150_stride = 1
        n_sections = 80
        tr2150_start_ix = 0
        tr2150_end_ix = tr2150_window_size
        
        tr100_window_size = int((n_tp_tr100 / n_tp_tr2150) * tr2150_window_size)
        tr100_stride = n_tp_tr100 // n_tp_tr2150
        tr100_start_ix = 0
        tr100_end_ix = tr100_window_size


        for j in range(n_sections):
            window_ix = i * n_sections * 2 + j * 2
            
            tr100_section = tr100[:, tr100_start_ix:tr100_end_ix]
            tr100_section_noise = tr100_noise[:, tr100_start_ix:tr100_end_ix]

            tr2150_section = tr2150[:, tr2150_start_ix:tr2150_end_ix]
            tr2150_section_noise = tr2150_noise[:, tr2150_start_ix:tr2150_end_ix]






            tr100_fnc_triu = np.corrcoef(tr100_section)[np.triu_indices(n_regions)]
            tr100_noise_fnc_triu = np.corrcoef(tr100_section_noise)[np.triu_indices(n_regions)]   #TODO: debug

            tr2150_fnc_triu = np.corrcoef(tr2150_section)[np.triu_indices(n_regions)]
            tr2150_noise_fnc_triu = np.corrcoef(tr2150_section_noise)[np.triu_indices(n_regions)]

            concat_tr100_tr2150 = np.concatenate((tr100_fnc_triu , tr2150_fnc_triu))
            concat_tr100_tr2150_noise = np.concatenate((tr100_noise_fnc_triu , tr2150_noise_fnc_triu))

            add_tr100_tr2150 = tr100_fnc_triu + tr2150_fnc_triu
            add_tr100_tr2150_noise = tr100_noise_fnc_triu + tr2150_noise_fnc_triu


            tr100_data.append({'subject'          : subject, 
                            'TR100ms_Window'   : tr100_fnc_triu, 
                            'target'           : '0'})
            tr100_data.append({'subject'          : subject, 
                            'TR100ms_Window'   : tr100_noise_fnc_triu, 
                            'target'           : '1'})
            
            tr2150_data.append({'subject'         : subject,
                                'TR2150ms_Window' : tr2150_fnc_triu, 
                                'target'          : '0'})
            tr2150_data.append({'subject'         : subject,
                                'TR2150ms_Window' : tr2150_noise_fnc_triu, 
                                'target'          : '1'})
            
            concat_data.append({'subject'          : subject, 
                                'Concat_Window'   : concat_tr100_tr2150,
                            'target'            : '0'})
            concat_data.append({'subject'          : subject, 
                                'Concat_Window'    : concat_tr100_tr2150_noise,
                                'target'           : '1'})
            
            add_data.append({'subject'             : subject,
                            'Add_Window'         : add_tr100_tr2150,
                            'target'             : '0'})
            add_data.append({'subject'             : subject,
                            'Add_Window'         : add_tr100_tr2150_noise,
                            'target'             : '1'})
            
            tr100_start_ix += tr100_stride
            tr100_end_ix = tr100_end_ix + tr100_stride
                
            tr2150_start_ix += tr2150_stride
            tr2150_end_ix = tr2150_end_ix + tr2150_stride



    tr100_df = pd.DataFrame(tr100_data)
    tr2150_df = pd.DataFrame(tr2150_data)
    concat_df = pd.DataFrame(concat_data)
    add_df = pd.DataFrame(add_data)


    logging.info(f'\n\n\n\n START SVM FOR snr {SNR}')
    
    
    group_tr100 = tr100_df['subject']
    y_tr100 = tr100_df['target']
    y_tr100 = np.array([str(entry) for entry in y_tr100])
    X_tr100 = tr100_df['TR100ms_Window']
    X_tr100 = np.array([np.array(entry) for entry in X_tr100])
    
    
    group_tr2150 = tr2150_df['subject']
    y_tr2150 = tr2150_df['target']
    y_tr2150 = np.array([str(entry) for entry in y_tr2150])
    X_tr2150 = tr2150_df['TR2150ms_Window']
    X_tr2150 = np.array([np.array(entry) for entry in X_tr2150])

    
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


    res1 = []
    res2 = []
    res3 = []
    res4 = []

    sgkf = StratifiedGroupKFold(n_splits=n_folds)

    for fold_number, (train_index, test_index) in enumerate(sgkf.split(X_tr100, y_tr100, group_tr100), start=1):
        sr1_train, sr1_test = X_tr100[train_index], X_tr100[test_index]
        sr2_train, sr2_test = X_tr2150[train_index], X_tr2150[test_index]
        add_train, add_test = X_add[train_index], X_add[test_index]
        concat_train, concat_test = X_concat[train_index], X_concat[test_index]

        y_train_tr100, y_test_tr100 = np.array(y_tr100)[train_index], np.array(y_tr100)[test_index]
        y_train_tr2150, y_test_tr2150 = np.array(y_tr2150)[train_index], np.array(y_tr2150)[test_index]
        y_train_add, y_test_add = np.array(y_add)[train_index], np.array(y_add)[test_index]
        y_train_concat, y_test_concat = np.array(y_concat)[train_index], np.array(y_concat)[test_index]

        svm_sr1 = SVC(probability=True, kernel='linear')
        svm_sr2 = SVC(probability=True, kernel='linear')
        svm_add = SVC(probability=True, kernel='linear')
        svm_concat = SVC(probability=True, kernel='linear')

        svm_sr1.fit(sr1_train.reshape(len(sr1_train), -1), y_train_tr100)
        svm_sr2.fit(sr2_train.reshape(len(sr2_train), -1), y_train_tr2150)
        svm_add.fit(add_train.reshape(len(add_train), -1), y_train_add)
        svm_concat.fit(concat_train.reshape(len(concat_train), -1), y_train_concat)

        fold_roc_sr1 = roc_auc_score(y_test_tr100, svm_sr1.predict_proba(sr1_test.reshape(len(sr1_test), -1))[:, 1])
        fold_roc_sr2 = roc_auc_score(y_test_tr2150, svm_sr2.predict_proba(sr2_test.reshape(len(sr2_test), -1))[:, 1])
        fold_roc_add = roc_auc_score(y_test_add, svm_add.predict_proba(add_test.reshape(len(add_test), -1))[:, 1])
        fold_roc_concat = roc_auc_score(y_test_concat, svm_concat.predict_proba(concat_test.reshape(len(concat_test), -1))[:, 1])


        res1.append({'snr'                 : SNR,
                        'fold'             : fold_number, 
                        'roc'              : fold_roc_sr1,
                        'sampling_rate'   : 'TR100'
                    })
        
        res2.append({'snr'                 : SNR,
                        'fold'             : fold_number, 
                        'roc'              : fold_roc_sr2,
                        'sampling_rate'   : 'TR2150'
                    })
        
        res3.append({'snr'                 : SNR,
                        'fold'             : fold_number, 
                        'roc'              : fold_roc_add,
                        'sampling_rate'   : 'Add'
                    })
        
        res4.append({'snr'                 : SNR,
                        'fold'             : fold_number, 
                        'roc'              : fold_roc_concat,
                        'sampling_rate'   : 'Concat'
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
    df1.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr1_{SNR}_nopoly.pkl')
    df2 = pd.DataFrame(res2)
    df2.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr2_{SNR}_nopoly.pkl')
    df3 = pd.DataFrame(res3)
    df3.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/concat_{SNR}_nopoly.pkl')
    df4 = pd.DataFrame(res4)
    df4.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/add_{SNR}_nopoly.pkl')