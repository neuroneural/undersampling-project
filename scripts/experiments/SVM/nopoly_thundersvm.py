import logging
import argparse
import pandas as pd
import numpy as np
import scipy.io
from scipy.stats import zscore
from scipy.signal import detrend
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import pickle
from thundersvm import SVC





def scale_noise(n, x, SNR):
    assert x.shape[0] == 53, 'timecourse dimension 0 should be 53'
    assert n.shape[0] == 53, 'noise dimension 0 should be 53'
    xTx = np.sum(np.square(x))
    nTn = np.sum(np.square(n))
    c = ((xTx / nTn)**0.5) / (10**(SNR/2)) 
    scaled_noise = c * n
    return scaled_noise


def create_colored_noise(cov_mat, L, noise_size):
    assert cov_mat.shape == (53, 53), 'cov_mat should be 53 x 53 matrix'
    assert L.shape == (53, 53), 'L should be 53 x 53 matrix'
    mean = np.zeros(cov_mat.shape[0])
    white_noise = np.random.multivariate_normal(mean, np.eye(cov_mat.shape[0]), size=noise_size)
    colored_noise = white_noise @ L.T
    colored_noise = colored_noise.T
    colored_noise = zscore(colored_noise, axis=1)
    colored_noise = detrend(colored_noise, axis=1)
    return colored_noise


def preprocess_timecourse(tc_data):
    assert tc_data.shape[0] == 53, 'timecourse dimension 0 should be 53'
    data = zscore(tc_data, axis=1)                       
    data = detrend(data, axis=1)         
    max_magnitudes = np.max(np.abs(data), axis=1, keepdims=True) 
    data = data / max_magnitudes
    return data
    

def parse_X_y_groups(data_df, name):
    group = data_df['subject']
    y = data_df['target']
    y = np.array([str(entry) for entry in y])
    X = data_df[f'{name}_Window']
    X = np.array([np.array(entry) for entry in X])
    return X, y, group



def main():
    project_dir = '/data/users2/jwardell1/undersampling-project'


    parser = argparse.ArgumentParser()

    
    parser.add_argument('-n', '--noise-dataset', type=str, help='noise dataset name', required=True)
    parser.add_argument('-s', '--signal-dataset', type=str, help='signal dataset name', required=True)
    parser.add_argument('-k', '--kernel-type', type=str, choices=['linear', 'rbf'], help='type of SVM kernel', required=True)

    parser.add_argument('-i', '--snr-int', type=float, nargs='+', help='upper, lower, step of SNR interval', required=False)
    parser.add_argument('-f', '--n-folds', type=int, help='number of folds for cross-validation', required=False)
    parser.add_argument('-nn', '--num_noise', type=int, help='number of noise iterations', required=False)
    parser.add_argument('-v', '--verbose', type=bool, help='turn on debug logging', required=False)
    
    args = parser.parse_args()

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
    n_folds = args.n_folds if args.n_folds != None else 7
    kernel_type = args.kernel_type
    num_noise = args.num_noise if args.num_noise != None else 1
    log_level = 'DEBUG' if args.verbose else 'INFO'

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Print the parsed arguments to verify
    logging.info(f'Noise Interval: {SNRs}')
    logging.info(f'Noise Dataset: {noise_dataset}')
    logging.info(f'Signal Dataset: {signal_dataset}')
    logging.info(f'Number of Folds: {n_folds}')
    logging.info(f'Kernel Type: {kernel_type}')
    logging.info(f'Noise Iterations: {num_noise}')

    
    noise_data = scipy.io.loadmat(f'../../../assets/data/{noise_dataset}_data.mat')
    L = noise_data['L']
    covariance_matrix = noise_data['cov_mat']

    if signal_dataset == 'OULU':
        undersampling_rate = 1
        NOISE_SIZE = 2961*2
    else: 
        NOISE_SIZE = 1200
        undersampling_rate = 6

    signal_data = pd.read_pickle(f'../../../assets/data/{signal_dataset}_data.pkl')

    subjects = np.unique(signal_data['subject'])


    for SNR in SNRs:
        res1 = []
        res2 = []
        res3 = []
        res4 = []


        for noise_ix in range(num_noise):
            noises = {}
            ################ loading and preprocessing
            all_data = []
            for subject in subjects:

                noises[subject] = create_colored_noise(covariance_matrix, L, NOISE_SIZE)
                logging.debug(f'computed noise for subject: {subject}')


                logging.debug(f'loading timecourse for subject {subject}')
                if signal_dataset == 'HCP': 
                    logging.debug('HCP dataset detected during loading')
                    sr1_tc = signal_data[
                        signal_data['subject'] == subject
                    ]['ica_timecourse'].iloc[0]
                

                else:
                    logging.debug('OULU dataset detected during loading')
                    sr1_tc = signal_data[
                        (signal_data['subject'] == subject) & 
                        (signal_data['sampling_rate'] == 'TR100')
                    ]['ica_timecourse'].iloc[0]

                    sr2_tc = signal_data[
                        (signal_data['subject'] == subject) & 
                        (signal_data['sampling_rate'] == 'TR2150')
                    ]['ica_timecourse'].iloc[0]


                sr1_tc = preprocess_timecourse(sr1_tc)
                sr2_tc = preprocess_timecourse(sr2_tc) if signal_dataset == 'OULU' else sr1_tc[:,::undersampling_rate]

                
                logging.debug(f'subject {subject} SR1 shape - {sr1_tc.shape}')
                logging.debug(f'subject {subject} SR2 shape - {sr2_tc.shape}')


                # sample from noise and scale
                noise_sr1 = None
                noise_sr2 = None

                if signal_dataset == 'HCP':
                    noise_sr1 = scale_noise(noises[subject], sr1_tc, SNR)
                    noise_sr2 = scale_noise(noises[subject][:,::undersampling_rate], sr2_tc, SNR)

                else:
                    noise_sr1 = scale_noise(noises[subject][:,::2], sr1_tc, SNR)
                    noise_sr2 = scale_noise(noises[subject][:,::33], sr2_tc, SNR)


                all_data.append(
                    {
                        'Subject_ID'             :  subject,
                        'SR1_Timecourse'         :  sr1_tc,
                        'SR2_Timecourse'         :  sr2_tc,
                        'SR1_Timecourse_Noise'   :  noise_sr1 + sr1_tc,
                        'SR2_Timecourse_Noise'   :  noise_sr2 + sr2_tc
                    }
                )
            ################ end loop over subjects



            data_df = pd.DataFrame(all_data)



            ################ windowing
            sr1_data = []
            sr2_data = []
            add_data = []
            concat_data = []
            for subject in subjects:
                logging.debug(f'begin windowing for subject {subject}')

                
                sr1 = data_df[data_df['Subject_ID'] == subject]['SR1_Timecourse'].iloc[0]
                sr1_noise = data_df[data_df['Subject_ID'] == subject]['SR1_Timecourse_Noise'].iloc[0]

                sr2 = data_df[data_df['Subject_ID'] == subject]['SR2_Timecourse'].iloc[0]
                sr2_noise = data_df[data_df['Subject_ID'] == subject]['SR2_Timecourse_Noise'].iloc[0]


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
                    sr1_noise_fnc_triu = np.corrcoef(sr1_section_noise)[np.triu_indices(n_regions)]

                    sr2_fnc_triu = np.corrcoef(sr2_section)[np.triu_indices(n_regions)]
                    sr2_noise_fnc_triu = np.corrcoef(sr2_section_noise)[np.triu_indices(n_regions)]

                    concat_sr1_sr2 = np.concatenate((sr1_fnc_triu , sr2_fnc_triu))
                    concat_sr1_sr2_noise = np.concatenate((sr1_noise_fnc_triu , sr2_noise_fnc_triu))

                    add_sr1_sr2 = sr1_fnc_triu + sr2_fnc_triu
                    add_sr1_sr2_noise = sr1_noise_fnc_triu + sr2_noise_fnc_triu


                    sr1_data.append({'subject': subject, 'SR1_Window': sr1_fnc_triu, 'target': '0'})
                    sr1_data.append({'subject': subject, 'SR1_Window': sr1_noise_fnc_triu, 'target': '1'})
                    

                    sr2_data.append({'subject': subject, 'SR2_Window': sr2_fnc_triu, 'target': '0'})
                    sr2_data.append({'subject': subject, 'SR2_Window': sr2_noise_fnc_triu, 'target': '1'})
                    

                    concat_data.append({'subject': subject,'Concat_Window': concat_sr1_sr2,'target': '0' })
                    concat_data.append({'subject': subject, 'Concat_Window': concat_sr1_sr2_noise,'target': '1'})
                    

                    add_data.append({'subject': subject,'Add_Window': add_sr1_sr2,'target': '0'})
                    add_data.append({'subject': subject,'Add_Window': add_sr1_sr2_noise,'target': '1'})


                    sr1_start_ix += sr1_stride
                    sr1_end_ix = sr1_start_ix + sr1_window_size
                        
                    sr2_start_ix += sr2_stride
                    sr2_end_ix = sr2_start_ix + sr2_window_size
                ################ end loop over window sections
            ################ end loop over subjects
            

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

            for name, X, y, group in datasets:
                for fold_ix, (train_index, test_index) in enumerate(sgkf.split(X, y, group), start=0):
                    fold_scores = []
                    _, X_test = X[train_index], X[test_index]
                    _, y_test = y[train_index], y[test_index]

                    # Load model weights and predict on test data
                    weights_dir = f'../../../assets/model_weights/{signal_dataset.lower()}/{kernel_type.lower()}'
                    model_filename = f'{name}_best_model_SNR_{SNR}_{kernel_type.upper()}.pkl'
                    model_path = f'{weights_dir}/{model_filename}'

                    with open(model_path, 'rb') as file:
                        svm = pickle.load(file)

                    y_pred = svm.predict(X_test)

                    fold_score = roc_auc_score(y_test, np.array(y_pred))
                    fold_scores.append(fold_score)
                    
                    logging.info(f' SNR {SNR} - {name} - fold {fold_ix} - noise iteration {noise_ix} fold_auc {fold_score}')
                    

                    results[name].append(
                        {
                            'snr'              : SNR,
                            'fold'             : fold_ix, 
                            'roc'              : fold_score,
                            'sampling_rate'    : name,
                        }
                    )
                    
                avg_roc = np.mean(fold_scores)
                logging.info(f'Average ROC AUC for {name}: {avg_roc}')


        pkl_dir = f'{project_dir}/{signal_dataset}/pkl-files' if project_dir != '.' else '.'

        for key, data in results.items():
            if data != []:
                df = pd.DataFrame(data)
                filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}_SVM_{kernel_type}.pkl'
                df.to_pickle(f'{pkl_dir}/{filename}')
                logging.info(f'saved results for {key} at {pkl_dir}/{filename}')

if __name__ == "__main__":
    main()

