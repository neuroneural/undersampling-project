import numpy as np
import pandas as pd
import sys

from polyssifier import poly_subject

import logging 

import scipy.sparse as sp
import scipy.io
from scipy.stats import zscore
import scipy.signal

from scipy.sparse.linalg import eigs
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import graph2adj

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state


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


def genData(A, rate=2, burnin=100, ssize=5000, nstd=0.1):
    Agt = A.copy()
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=nstd)
    data = data[:, burnin:]
    return data[:, ::rate]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#Step 0: Iterate through values for nstd, burnin, noise_svar
nstd = 1.0
burn = 100
threshold = 0.0001

num_noise = 3

NOISE_SIZE = 2961*2
NUM_SUBS = 10
subjects = [20150210, 20150417, 20150428, 20151110, 20151127, 
            20150410, 20150421, 20151030, 20151117, 20151204]


res1 = []
res2 = []
res3 = []
res4 = []


if len(sys.argv) != 4:
    print("Usage: python poly_noise1.py SNR graph_dir graph_ix")
    sys.exit(1)

SNR = float(sys.argv[1])
graph_dir = sys.argv[2]
graph_ix = int(sys.argv[3])
g = np.load(graph_dir, allow_pickle=True)
'''
SNR = 2
graph_ix = 1000
g = gk.ringmore(53, 10)
'''


scalar = 10**(SNR/-2)

logging.info(f'\t\t\t\tSNR- {SNR}')
logging.info(f'\t\t\t\tscalar- {scalar}')
logging.info(f'\t\t\t\tGRAPH IX- {graph_ix}')

num_converged = 0
noises = dict()
converged_subjects = []





A = graph2adj(g)
u_rate = 1


#Using the graphs, generate a number of noise matrices for all subjects until converged
for noise_ix in range(num_noise):
    while num_converged < len(subjects):
        for subject in range(len(subjects)):
            if subject in converged_subjects:
                continue  


            try:
                W = create_stable_weighted_matrix(A, threshold=0.001, powers=[2])
                var_noise = genData(W, rate=u_rate, burnin=burn, ssize=NOISE_SIZE, nstd=nstd)
                var_noise = zscore(var_noise, axis=1)
                noises[subjects[subject]] = var_noise*scalar
                num_converged += 1
                converged_subjects.append(subject)
            
            except Exception as e:
                print(f'Convergence error while generating matrix for dir {subjects[subject]}, num converged: {num_converged}')
                print(e)
                continue
        


    tc_sr1 = dict()
    tc_sr2 = dict()
    tc_sr1_noise = dict()
    tc_sr2_noise = dict()
    with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/allsubs_TCs.txt', 'r') as tc_data:
        lines = tc_data.readlines()

    for i in range(0, len(lines), 2):
        file_path_sr1 = lines[i].strip()
        file_path_sr2 = lines[i + 1].strip()
        logging.info(f'Processing SR1: {file_path_sr1}')
        logging.info(f'Processing SR2: {file_path_sr2}')
        try:
            sr1 = scipy.io.loadmat(file_path_sr1)['TCMax']
            sr2 = scipy.io.loadmat(file_path_sr2)['TCMax']
        except:
            continue

        if sr1.shape[0] != 53:
            sr1 = sr1.T

        if sr2.shape[0] != 53:
            sr2 = sr2.T

        if sr1.shape[1] < sr2.shape[1]:
            temp = sr2
            sr2 = sr1
            sr1 = temp

        logging.info(f'sr1.shape - {sr1.shape}')
        logging.info(f'sr2.shape - {sr2.shape}')

        #zscore tc_data
        sr1 = zscore(sr1, axis=1)
        sr2 = zscore(sr2, axis=1)

        sr1 = scipy.signal.detrend(sr1, axis=1)
        sr2 = scipy.signal.detrend(sr2, axis=1)
        
        var_noise = noises[subjects[i//2]]

        tc_sr1[subjects[i//2]] = sr1 #TR=100ms
        tc_sr2[subjects[i//2]] = sr2 #TR=2150ms

        tc_sr1_noise[subjects[i//2]] = sr1 + var_noise[:,::2]
        tc_sr2_noise[subjects[i//2]] = sr2 + var_noise[:,::33] 

    #TODO- Step 2: Perform windowing on noise/non-noise data
    windows_sr1 = []
    windows_sr2 = []
    windows_concat = []
    windows_add = []

    for i in range(NUM_SUBS):
        subject_id = subjects[i]
        if subject_id == '': 
            continue

        sr1 = tc_sr1[subject_id]#TR=100ms
        sr1_noise = tc_sr1_noise[subject_id]
        
        sr2 = tc_sr2[subject_id]#TR=2150ms
        sr2_noise = tc_sr2_noise[subject_id]

        n_regions, n_tp_tr100 = sr1.shape
        _, n_tp_tr2150 = sr2.shape

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
            logging.info(f"Processing section {j+1}/{n_sections} for subject {subject_id}")

            tr100_section = sr1[:, tr100_start_ix:tr100_end_ix]
            tr100_section_noise = sr1_noise[:, tr100_start_ix:tr100_end_ix]



            tr2150_section = sr2[:, tr2150_start_ix:tr2150_end_ix]
            tr2150_section_noise = sr2_noise[:, tr2150_start_ix:tr2150_end_ix]


            windows_sr1.insert(j, (np.corrcoef(tr100_section)[np.triu_indices(n_regions)], 0, subject_id))
            windows_sr1.insert(j+1, (np.corrcoef(tr100_section_noise)[np.triu_indices(n_regions)], 1, subject_id))


            windows_sr2.insert(j, (np.corrcoef(tr2150_section)[np.triu_indices(n_regions)], 0, subject_id))
            windows_sr2.insert(j+1, (np.corrcoef(tr2150_section_noise)[np.triu_indices(n_regions)], 1, subject_id))


            windows_concat.insert(j, (np.concatenate((windows_sr1[j][0], windows_sr2[j][0])), 0, subject_id))
            windows_concat.insert(j+1, (np.concatenate((windows_sr1[j+1][0], windows_sr2[j+1][0])), 1, subject_id))
            
            windows_add.insert(j, (windows_sr1[j][0]+windows_sr2[j][0], 0, subject_id))
            windows_add.insert(j+1, (windows_sr1[j+1][0]+windows_sr2[j+1][0], 1, subject_id))

            tr100_start_ix += tr100_stride
            tr100_end_ix = tr100_end_ix + tr100_stride
                
            tr2150_start_ix += tr2150_stride
            tr2150_end_ix = tr2150_end_ix + tr2150_stride

    del tc_sr1
    del tc_sr2
    del tc_sr1_noise
    del tc_sr2_noise


    #TODO- Step 3: Run polyssifier on windowed data
    data_sr1 = [entry[0] for entry in windows_sr1]
    labels_sr1 = [entry[1] for entry in windows_sr1]
    groups_sr1 = [entry[2] for entry in windows_sr1]

    data_sr2 = [entry[0] for entry in windows_sr2]
    labels_sr2 = [entry[1] for entry in windows_sr2]
    groups_sr2 = [entry[2] for entry in windows_sr2]

    data_concat = [entry[0] for entry in windows_concat]
    labels_concat = [entry[1] for entry in windows_concat]
    groups_concat = [entry[2] for entry in windows_concat]

    data_add = [entry[0] for entry in windows_add]
    labels_add = [entry[1] for entry in windows_add]
    groups_add = [entry[2] for entry in windows_add]
    from sklearn.impute import SimpleImputer


    # Initialize SimpleImputer
    imputer = SimpleImputer(strategy='mean')

    # Impute missing values and encode labels for each dataset
    data_sr1 = imputer.fit_transform(data_sr1)
    data_sr2 = imputer.fit_transform(data_sr2)
    data_concat = imputer.fit_transform(data_concat)
    data_add = imputer.fit_transform(data_add)


    # Set a global seed for numpy operations
    np.random.seed(42)

    # When creating a random state for sklearn operations, use:
    random_state = check_random_state(42)

    scaler1 = MinMaxScaler()#StandardScaler()
    data_scaled1 = scaler1.fit_transform(data_sr1)
    # Perform poly_subject and plot_scores for each dataset
    report1 = poly_subject(data_scaled1, np.array(labels_sr1), groups_sr1, n_folds=10, random_state=random_state,
                            project_name=f'SR1_noise_{scalar}', scale=True, 
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    for classifier in report1.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res1.append({'graph_no': graph_ix,
                        'nstd': nstd,
                        'burnin': burn,
                        'noise_no': noise_ix,
                        'snr': SNR,
                        'scalar': scalar,
                        'classifier': classifier,
                        'test_scores': report1.scores[classifier, 'test'], 
                        'target': report1.target, 
                        'predictions': np.array(report1.predictions[classifier]).astype(int),
                        'test_proba': report1.test_proba[classifier]})

        logging.info(report1.scores[classifier, 'test'])

    scaler2 = MinMaxScaler()#StandardScaler()
    data_scaled2 = scaler2.fit_transform(data_sr2)
    report2 = poly_subject(data_scaled2, np.array(labels_sr2), groups_sr2, n_folds=10, random_state=random_state,
                            project_name=f'SR2_noise_{scalar}', scale=True, 
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    for classifier in report2.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res2.append({'graph_no': graph_ix,
                    'nstd': nstd,
                    'burnin': burn,
                    'noise_no': noise_ix,
                    'snr': SNR,
                    'scalar': scalar,
                    'classifier': classifier,
                    'test_scores': report2.scores[classifier, 'test'], 
                    'target': report2.target, 
                    'predictions': np.array(report2.predictions[classifier]).astype(int), 
                    'test_proba': report2.test_proba[classifier]})
        
        logging.info(report2.scores[classifier, 'test'])

    scaler3 = MinMaxScaler()#StandardScaler()
    data_scaled3 = scaler3.fit_transform(data_concat)
    report3 = poly_subject(data_scaled3, np.array(labels_concat), groups_concat, n_folds=10, random_state=random_state,
                            project_name=f'CONCAT_noise_{scalar}', scale=True, 
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    for classifier in report3.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res3.append({'graph_no': graph_ix,
                    'nstd': nstd,
                    'burnin': burn,
                    'noise_no': noise_ix,
                    'snr': SNR,
                    'scalar': scalar,
                    'classifier': classifier,
                    'test_scores': report3.scores[classifier, 'test'], 
                    'target': report3.target, 
                    'predictions': np.array(report3.predictions[classifier]).astype(int), 
                    'test_proba': report3.test_proba[classifier]})
        
        logging.info(report3.scores[classifier, 'test'])
        
    scaler4 = MinMaxScaler()#StandardScaler()
    data_scaled4 = scaler4.fit_transform(data_add)
    report4 = poly_subject(data_scaled4, np.array(labels_add), groups_add, n_folds=10, random_state=random_state,
                        project_name=f'ADD_noise_{scalar}', scale=True, 
                        exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    for classifier in report4.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res4.append({'graph_no': graph_ix,
                    'nstd': nstd,
                    'burnin': burn,
                    'noise_no': noise_ix,
                    'snr': SNR,
                    'scalar': scalar,
                    'classifier': classifier,
                    'test_scores': report4.scores[classifier, 'test'], 
                    'target': report4.target, 
                    'predictions': np.array(report4.predictions[classifier]).astype(int), 
                    'test_proba': report4.test_proba[classifier]})


        logging.info(report4.scores[classifier, 'test'])


#populate dataframe here for combimation of noise
df1 = pd.DataFrame(res1)
df2 = pd.DataFrame(res2)
df3 = pd.DataFrame(res3)
df4 = pd.DataFrame(res4)
df1.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr1_{SNR}_{graph_ix}.pkl')
df2.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr2_{SNR}_{graph_ix}.pkl')
df3.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/concat_{SNR}_{graph_ix}.pkl')
df4.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/add_{SNR}_{graph_ix}.pkl')

