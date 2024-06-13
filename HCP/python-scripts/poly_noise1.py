
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


import numpy as np
import pandas as pd
import sys

from polyssifier import poly

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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.impute import SimpleImputer


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def check_matrix_powers(W, A, powers, threshold):
    for n in powers:
        W_n = np.linalg.matrix_power(W, n)
        non_zero_indices = np.nonzero(W_n)
        if (np.abs(W_n[non_zero_indices]) < threshold).any():
            return False
    return True


def create_stable_weighted_matrix(
    A,
    threshold=0.1,
    powers=[1, 2, 3, 4],
    max_attempts=1000,
    damping_factor=0.99,
    random_state=None,
):
    np.random.seed(
        random_state
    )  # Set random seed for reproducibility if provided
    attempts = 0

    while attempts < max_attempts:
        # Generate a random matrix with the same sparsity pattern as A
        random_weights = np.random.randn(*A.shape)
        weighted_matrix = A * random_weights

        # Convert to sparse format for efficient eigenvalue computation
        weighted_sparse = sp.csr_matrix(weighted_matrix)

        # Compute the largest eigenvalue in magnitude
        eigenvalues, _ = eigs(weighted_sparse, k=1, which="LM")
        max_eigenvalue = np.abs(eigenvalues[0])

        # Scale the matrix so that the spectral radius is slightly less than 1
        if max_eigenvalue > 0:
            weighted_matrix *= damping_factor / max_eigenvalue
            # Check if the powers of the matrix preserve the threshold for non-zero entries of A
            if check_matrix_powers(weighted_matrix, A, powers, threshold):
                return weighted_matrix

        attempts += 1

    raise ValueError(
        f"Unable to create a matrix satisfying the condition after {max_attempts} attempts."
    )


def drawsamplesLG(A, nstd, samples):
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data


def genData(A, rate=2, burnin=100, ssize=5000, nstd=1):
    Agt = A.copy()
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=nstd)
    data = data[:, burnin:]
    return data[:, ::rate]





if len(sys.argv) != 5:
    print("Usage: python poly_noise1.py SNR graph_dir graph_ix undersampling_factor")
    sys.exit(1)

SNR = float(sys.argv[1])
graph_dir = sys.argv[2]
graph_ix = int(sys.argv[3])
undersampling_factor = int(sys.argv[4])

"""
SNRs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SNR = 1
graph_ix = 1002
graph_dir = '/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/g4.pkl'
undersampling_factor = 3
"""

g = np.load(graph_dir, allow_pickle=True)
A = graph2adj(g)
u_rate = 1



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


nstd = 1.0
burn = 100
threshold = 0.0001

np.random.seed(42)
NOISE_SIZE = 1200
subjects = np.loadtxt("/data/users2/jwardell1/undersampling-project/HCP/txt-files/subjects.txt", dtype=str)
#rand_sub_ix = np.random.choice(len(subjects), 10)
#subjects = subjects[rand_sub_ix]
NUM_SUBS = len(subjects)

num_graphs = 3
num_noise = 5
n_folds = 10
n_threads= 40


logging.info(f'\t\t\t\tGraph Number {graph_ix} of {num_graphs}')
logging.info(f'\t\t\t\tNUM_SUBS {NUM_SUBS}')





#Using the loaded graph, generate a number of noise matrices for all subjects until converged
for noise_ix in range(num_noise):
    #for SNR in SNRs:
    num_converged = 0
    converged_subjects = []
    noises = dict()
    while num_converged < NUM_SUBS:
        for subject in subjects:
            if subject in converged_subjects:
                continue

            try:
                W = create_stable_weighted_matrix(A, threshold=threshold, powers=[2])
                var_noise = genData(W, rate=u_rate, burnin=burn, ssize=NOISE_SIZE, nstd=nstd)
                var_noise = zscore(var_noise, axis=1)
                noises[subject] = var_noise 
                num_converged += 1
                converged_subjects.append(subject)

            except Exception as e:
                print(e)
                logging.info(f'num converged: {num_converged}')
            



    
    all_data = []

    #tc_filepath = '/data/users2/jwardell1/undersampling-project/HCP/txt-files/tc_data.txt'

    #with open(tc_filepath, 'r') as tc_data:
    #    lines = np.array(tc_data.readlines())

    #lines = lines[rand_sub_ix]

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

        sr2_tc_zc_dt = sr1_tc_zc_dt[:,::undersampling_factor]
        logging.info(f'sr2.shape - {sr2_tc_zc_dt.shape}')

        sr1_tc_zc_dt = MinMaxScaler(feature_range=(-1,1)).fit_transform(sr1_tc_zc_dt)
        sr2_tc_zc_dt = MinMaxScaler(feature_range=(-1,1)).fit_transform(sr2_tc_zc_dt)


        noise_sr1 = noises[subject]
        noise_sr2 = noises[subject][:,::undersampling_factor]

        all_data.append({'Subject_ID'             : str(subject), 
                        'VAR_Noise'               : noises[subject], 
                        'SR1_Noise'               : noise_sr1, 
                        'SR2_Noise'               : noise_sr2, 
                        'SR1_Timecourse'          : sr1_tc_zc_dt, 
                        'SR2_Timecourse'          : sr2_tc_zc_dt
                        })
        
    data_df = pd.DataFrame(all_data)

    xTx_sr1 = np.sum(np.square(data_df['SR1_Timecourse'].mean()))
    nTn_sr1 = np.sum(np.square(data_df['SR1_Noise'].mean()))
    scalar_sr1 = ((xTx_sr1 / nTn_sr1)**0.5) / (10**(SNR/2))

    xTx_sr2 = np.sum(np.square(data_df['SR2_Timecourse'].mean()))
    nTn_sr2 = np.sum(np.square(data_df['SR2_Noise'].mean()))
    scalar_sr2 = ((xTx_sr2 / nTn_sr2)**0.5) / (10**(SNR/2))


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
    
    for subject in subjects:
        sub_row = data_df[data_df['Subject_ID']  == subject]
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
            window_ix = i * n_sections * 2 + j * 2
            
            sr1_section = sr1[:, sr1_start_ix:sr1_end_ix]
            sr1_section_noise = sr1_noise[:, sr1_start_ix:sr1_end_ix]

            sr2_section = sr2[:, sr2_start_ix:sr2_end_ix]
            sr2_section_noise = sr2_noise[:, sr2_start_ix:sr2_end_ix]






            sr1_fnc_triu = np.corrcoef(sr1_section)[np.triu_indices(n_regions)]
            sr1_noise_fnc_triu = np.corrcoef(sr1_section_noise)[np.triu_indices(n_regions)]    #TODO: debugging

            sr2_fnc_triu = np.corrcoef(sr2_section)[np.triu_indices(n_regions)]
            sr2_noise_fnc_triu = np.corrcoef(sr2_section_noise)[np.triu_indices(n_regions)]

            concat_sr1_sr2 = np.concatenate((sr1_fnc_triu , sr2_fnc_triu))
            concat_sr1_sr2_noise = np.concatenate((sr1_noise_fnc_triu , sr2_noise_fnc_triu))

            add_sr1_sr2 = sr1_fnc_triu + sr2_fnc_triu
            add_sr1_sr2_noise = sr1_noise_fnc_triu + sr2_noise_fnc_triu


            sr1_data.append({'subject'          : subject, 
                                'SR1_Window'    : sr1_fnc_triu, 
                                'target'        : '0'})
            sr1_data.append({'subject'          : subject, 
                            'SR1_Window'        : sr1_noise_fnc_triu, 
                            'target'            : '1'})
            
            sr2_data.append({'subject'         : subject,
                                'SR2_Window'   : sr2_fnc_triu, 
                                'target'       : '0'})
            sr2_data.append({'subject'         : subject,
                                'SR2_Window'   : sr2_noise_fnc_triu, 
                                'target'       : '1'})
            
            concat_data.append({'subject'          : subject, 
                                'Concat_Window'    : concat_sr1_sr2,
                            'target'               : '0'})
            concat_data.append({'subject'          : subject, 
                                'Concat_Window'    : concat_sr1_sr2_noise,
                                'target'           : '1'})
            
            add_data.append({'subject'             : subject,
                            'Add_Window'           : add_sr1_sr2,
                            'target'               : '0'})
            add_data.append({'subject'             : subject,
                            'Add_Window'           : add_sr1_sr2_noise,
                            'target'               : '1'})
            
            sr1_start_ix += sr1_stride
            sr1_end_ix = sr1_end_ix + sr1_stride
                
            sr2_start_ix += sr2_stride
            sr2_end_ix = sr2_end_ix + sr2_stride



    sr1_df = pd.DataFrame(sr1_data)
    sr2_df = pd.DataFrame(sr2_data)
    concat_df = pd.DataFrame(concat_data)
    add_df = pd.DataFrame(add_data)




    #############################
    #   SR1
    #############################
    logging.info(f'\n\n\n\n START POLYSSIFIER FOR SR1 snr {SNR} noise_ix {noise_ix}')
    group_sr1 = sr1_df['subject']
    y_sr1 = sr1_df['target']
    y_sr1 = np.array([str(entry) for entry in y_sr1])
    X_sr1 = sr1_df['SR1_Window']
    X_sr1 = np.array([np.array(entry) for entry in X_sr1])

    res1 = []
    report1 = poly(data=X_sr1, label=y_sr1, groups=group_sr1, n_folds=n_folds, scale=True, concurrency=n_threads, save=False, 
                    exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')

    for classifier in report1.scores.columns.levels[0]:
                if classifier == 'Voting':
                    continue

                res1.append({'graph_no': graph_ix,
                                'nstd': nstd,
                                'burnin': burn,
                                'noise_no': noise_ix,
                                'snr': SNR,
                                'classifier': classifier,
                                'test_scores': report1.scores[classifier, 'test'], 
                                'target': report1.target, 
                                'predictions': np.array(report1.predictions[classifier]).astype(int),
                                'test_proba': report1.test_proba[classifier]})

                logging.info(report1.scores[classifier, 'test'])



    #############################
    #   SR2
    #############################
    logging.info(f'\n\n\n\n START POLYSSIFIER FOR SR2 snr {SNR} noise_ix {noise_ix}')
    group_sr2 = sr2_df['subject']
    y_sr2 = sr2_df['target']
    y_sr2 = np.array([str(entry) for entry in y_sr2])
    X_sr2 = sr2_df['SR2_Window']
    X_sr2 = np.array([np.array(entry) for entry in X_sr2])

    res2 = []
    report2 = poly(data=X_sr2, label=y_sr2, groups=group_sr2, n_folds=n_folds, scale=True, concurrency=n_threads, save=False, 
                    exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')

    for classifier in report2.scores.columns.levels[0]:                                                                                         # iterate through all classifiers in the report
                if classifier == 'Voting':
                    continue

                res2.append({'graph_no': graph_ix,                                                                                                      # save the SR1 results to a dict for results dataframe
                                'nstd': nstd,
                                'burnin': burn,
                                'noise_no': noise_ix,
                                'snr': SNR,
                                'classifier': classifier,
                                'test_scores': report2.scores[classifier, 'test'], 
                                'target': report2.target, 
                                'predictions': np.array(report2.predictions[classifier]).astype(int),
                                'test_proba': report2.test_proba[classifier]})

                logging.info(report2.scores[classifier, 'test'])



    #############################
    #   CONCAT
    #############################
    logging.info(f'\n\n\n\n START POLYSSIFIER FOR CONCAT snr {SNR} noise_ix {noise_ix}')
    group_concat = concat_df['subject']
    y_concat = concat_df['target']
    y_concat = np.array([str(entry) for entry in y_concat])
    X_concat = concat_df['Concat_Window']
    X_concat = np.array([np.array(entry) for entry in X_concat])

    res3 = []
    report3 = poly(data=X_concat, label=y_concat, groups=group_concat, n_folds=n_folds, scale=True, concurrency=n_threads, save=False, 
                    exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')

    for classifier in report3.scores.columns.levels[0]:
                if classifier == 'Voting':
                    continue

                res3.append({'graph_no': graph_ix,
                                'nstd': nstd,
                                'burnin': burn,
                                'noise_no': noise_ix,
                                'snr': SNR,
                                'classifier': classifier,
                                'test_scores': report3.scores[classifier, 'test'], 
                                'target': report3.target, 
                                'predictions': np.array(report3.predictions[classifier]).astype(int),
                                'test_proba': report3.test_proba[classifier]})

                logging.info(report3.scores[classifier, 'test'])


    #############################
    #   ADD
    #############################
    logging.info(f'\n\n\n\n START POLYSSIFIER FOR ADD snr {SNR} noise_ix {noise_ix}')
    group_add = add_df['subject']
    y_add = add_df['target']
    y_add = np.array([str(entry) for entry in y_add])
    X_add = add_df['Add_Window']
    X_add = np.array([np.array(entry) for entry in X_add])

    res4 = []
    report4 = poly(data=X_add, label=y_add, groups=group_add, n_folds=n_folds, scale=True, concurrency=n_threads, save=False, 
                    exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')

    for classifier in report4.scores.columns.levels[0]:                                                                                         # iterate through all classifiers in the report
                if classifier == 'Voting':
                    continue

                res4.append({'graph_no': graph_ix,                                                                                                      # save the SR1 results to a dict for results dataframe
                                'nstd': nstd,
                                'burnin': burn,
                                'noise_no': noise_ix,
                                'snr': SNR,
                                'classifier': classifier,
                                'test_scores': report4.scores[classifier, 'test'], 
                                'target': report4.target, 
                                'predictions': np.array(report4.predictions[classifier]).astype(int),
                                'test_proba': report4.test_proba[classifier]})

                logging.info(report4.scores[classifier, 'test'])




df1 = pd.DataFrame(res1)  
df1.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/us-{undersampling_factor}/sr1_{SNR}_{graph_ix}.pkl')
df2 = pd.DataFrame(res2)
df2.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/us-{undersampling_factor}/sr2_{SNR}_{graph_ix}.pkl')
df3 = pd.DataFrame(res3)
df3.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/us-{undersampling_factor}/concat_{SNR}_{graph_ix}.pkl')
df4 = pd.DataFrame(res4)
df4.to_pickle(f'/data/users2/jwardell1/undersampling-project/HCP/pkl-files/us-{undersampling_factor}/add_{SNR}_{graph_ix}.pkl')