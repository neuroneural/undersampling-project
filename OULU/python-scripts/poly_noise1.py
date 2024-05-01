import numpy as np
import pandas as pd
import sys

from polyssifier import poly

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
from sklearn.impute import SimpleImputer


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

num_graphs = 3
num_noise = 3
n_folds = 10
n_threads= 32



NOISE_SIZE = 2961*2
NUM_SUBS = 10
subjects = [20150210, 20150417, 20150428, 20151110, 20151127, 
            20150410, 20150421, 20151030, 20151117, 20151204]


res1 = []
res2 = []
res3 = []
res4 = []
'''
if len(sys.argv) != 4:
    print("Usage: python poly_noise1.py SNR graph_dir graph_ix")
    sys.exit(1)

SNR = float(sys.argv[1])                                  # signal to noise ratio
graph_dir = sys.argv[2]                                   # directory of pre-generated graph
graph_ix = int(sys.argv[3])                               # graph number
g = np.load(graph_dir, allow_pickle=True)                 # load graph

'''
SNR = 1                                           
graph_ix = 1000
graph_dir = '/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/g0.pkl'
g = np.load(graph_dir, allow_pickle=True)          #= gk.ringmore(53, 10)  #graph from which noise matrix is generated





scalar = 10**(SNR/-2)                                     # number to multiply noise by for a given SNR

logging.info(f'\t\t\t\t SNR- {SNR}')
logging.info(f'\t\t\t\t scalar- {scalar}')
logging.info(f'\t\t\t\t GRAPH IX- {graph_ix}')






A = graph2adj(g)
u_rate = 1
logging.info(f'\t\t\t\tGraph Number {graph_ix} of {num_graphs}')

#Using the graphs, generate a number of noise matrices for all subjects until converged

for noise_ix in range(num_noise):                   # run polyssifier for a number of noise samples
    logging.info(f'\t\t\t\t NOISE IX- {noise_ix}')
    num_converged = 0                               # counter for how many subjects' noise matrices have converged
    noises = dict()                                 # dict for noises.  key: sub_id , value: sub's noise matrix
    converged_subjects = []                         # holds IDs of converged subjects

    while num_converged < len(subjects):            # iterate until all subjects are converged
        for subject in subjects:                    # iterate through all subjects
            if subject in converged_subjects:       # skip converged subjects
                continue  


            try:                                                                                        # try to generate a noise sample
                W = create_stable_weighted_matrix(A, threshold=threshold, powers=[2])                   # use the ringmore graph to get a matrix W
                var_noise = genData(W, rate=u_rate, burnin=burn, ssize=NOISE_SIZE, nstd=nstd)           # use W to generate the noise sample
                var_noise = zscore(var_noise, axis=1)                                                   # z-score the noise
                noises[subject] = var_noise*scalar                                                      # scale the noise to the SNR and store in the noises dict. key: subid , value: noisemat
                num_converged += 1                                                                      # increment the num_converged counter
                converged_subjects.append(subject)                                                      # add the sub id to the list of converged subs
            
            except Exception as e:                                                                      # catch convergence exception
                print(f'Convergence error while generating matrix for dir {subject}, num converged: {num_converged}')
                print(e)
                continue
        


    tc_sr1 = dict()         # dict for time courses                 key: subid , value: sub's TR=100ms timecourse
    tc_sr2 = dict()         # dict for time courses                 key: subid , value: sub's TR=2150ms timecourse
    tc_sr1_noise = dict()   # dict for noise corrupted timecourses  key: subid , value: sub's noisy TR=100ms timecourse
    tc_sr2_noise = dict()   # dict for noise corrupted timecourses  key: subid , value: sub's noisy TR=2150ms timecourse
    with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/allsubs_TCs.txt', 'r') as tc_data:
        lines = tc_data.readlines()

    for i in range(0, len(lines), 2):                     # iterate over every sub's two timecourse files (load TCs, zscore, detrend, add noise, & save)
        file_path_sr1 = lines[i].strip()                  # get first tc filepath
        file_path_sr2 = lines[i + 1].strip()              # get second tc filepath
        logging.info(f'Processing SR1: {file_path_sr1}')  
        logging.info(f'Processing SR2: {file_path_sr2}')
        try:
            sr1 = scipy.io.loadmat(file_path_sr1)['TCMax'] # load first timecourse
            sr2 = scipy.io.loadmat(file_path_sr2)['TCMax'] # load second timecourse
        except:
            continue

        if sr1.shape[0] != 53:           # make first tc dimensions #comp by #timepoints
            sr1 = sr1.T

        if sr2.shape[0] != 53:           # make second tc dimensions #comp by #timepoints
            sr2 = sr2.T

        if sr1.shape[1] < sr2.shape[1]:  # make sr1 TR=100ms and sr2 TR=2150ms
            temp = sr2
            sr2 = sr1
            sr1 = temp

        logging.info(f'sr1.shape - {sr1.shape}')
        logging.info(f'sr2.shape - {sr2.shape}')

        #zscore tc_data
        sr1_zs = zscore(sr1, axis=1)                                    # zscore TR=100ms timecourse
        sr2_zs = zscore(sr2, axis=1)                                    # zscore TR=2150ms timecourse

        sr1_zs_dt = scipy.signal.detrend(sr1_zs, axis=1)                # apply linear detrending on TR=100ms  timecourse
        sr2_zs_dt = scipy.signal.detrend(sr2_zs, axis=1)                # apply linear detrending on TR=2150ms timecourse
        
        var_noise = noises[subjects[i//2]]                              # load the noise matrix for current sub from noises dict

        tc_sr1[subjects[i//2]] = sr1_zs_dt                              # save the sub's zscored, detrended TR=100ms  TC to sr1 TCs dict
        tc_sr2[subjects[i//2]] = sr2_zs_dt                              # save the sub's zscored, detrended TR=2150ms TC to sr2 TCs dict

        tc_sr1_noise[subjects[i//2]] = sr1_zs_dt + var_noise[:,::2]     # sample from sub's noise matrix, add noise to TR=100ms  TC, & save to sr1 noisy TCs dict
        tc_sr2_noise[subjects[i//2]] = sr2_zs_dt + var_noise[:,::33]    # sample from sub's noise matrix, add noise to TR=2150ms TC, & save to sr2 noisy TCs dict




    windows_sr1 = []                            # list to hold windowed TR=100ms  TC data for all subjects      -> 3-tuple (window FNC triu entries, sub id, label 0: no noise, 1: noise)
    windows_sr2 = []                            # list to hold windowed TR=2150ms TC data for all subjects      -> 3-tuple (window FNC triu entries, sub id, label 0: no noise, 1: noise)
    windows_concat = []                         # list to hold concatenated windowed TC data for all subjects   -> 3-tuple (window FNC triu entries, sub id, label 0: no noise, 1: noise)
    windows_add = []                            # list to hold added windowed TC data for all subjects          -> 3-tuple (window FNC triu entries, sub id, label 0: no noise, 1: noise)

    
    for i in range(NUM_SUBS):                                                       # iterate over all subjects to load a sub's TCs (noise/none)
        subject_id = subjects[i]                                                    # set current subid
        if subject_id == '': 
            continue

        sr1 = tc_sr1[subject_id]                                                    # load current sub's TR=100ms  TC from dict
        sr1_noise = tc_sr1_noise[subject_id]                                        # load current sub's noisy TR=100ms  TC from dict
        
        sr2 = tc_sr2[subject_id]                                                    # load current sub's TR=2150ms TC from dict
        sr2_noise = tc_sr2_noise[subject_id]                                        # load current sub's noisy TR=2150ms TC from dict

        n_regions, n_tp_tr100 = sr1.shape                                           # set n_regions (components) and sr1 num time points from TR=100ms TC
        _, n_tp_tr2150 = sr2.shape                                                  # set sr2 num time points from TR=2150ms TC

        tr2150_window_size = 100                                                    # set window size for TR=2150ms windows
        tr2150_stride = 1                                                           # set stride for TR=2150ms windows
        n_sections = 80                                                             # set number of sections (windows) for TR=2150ms
        tr2150_start_ix = 0                                                         # set starting ix for TR=2150ms
        tr2150_end_ix = tr2150_window_size                                          # set end ix as end of first window (window size) for TR=2150ms

        tr100_window_size = int((n_tp_tr100 / n_tp_tr2150) * tr2150_window_size)    # set window size for TR=100ms using ratio ws_sr1/tp_sr1 = ws_sr2/tp_sr2
        tr100_stride = n_tp_tr100 // n_tp_tr2150                                    # set stride for TR=100ms as num tp SR1 for each SR2 tp
        tr100_start_ix = 0                                                          # set starting ix for TR=100ms
        tr100_end_ix = tr100_window_size                                            # set end ix as end of first window (window size) for TR=100ms




        for j in range(n_sections):                                                                                                     # for current sub's data, iterate over all windows "sections" (n_sections) as j
            window_ix = i * n_sections * 2 + j * 2                                                                                                             # index used for pulling stored sr1&2 windowed data
            logging.info(f"Processing section {j+1}/{n_sections} for subject {subject_id}")
            logging.info(f'window_ix {window_ix}')
            logging.info(f'j {j}')


            tr100_section = sr1[:, tr100_start_ix:tr100_end_ix]                                                                         # slice current sub's TR=100ms TC from startix to endix across all ICA comps
            tr100_section_noise = sr1_noise[:, tr100_start_ix:tr100_end_ix]                                                             # slice current sub's NOISY TR=100ms TC from startix to endix across all ICA comps     



            tr2150_section = sr2[:, tr2150_start_ix:tr2150_end_ix]                                                                      # slice current sub's TR=2150ms TC from startix to endix across all ICA comps
            tr2150_section_noise = sr2_noise[:, tr2150_start_ix:tr2150_end_ix]                                                          # slice current sub's NOISY TR=2150ms TC from startix to endix across all ICA comps     


            windows_sr1.append((np.corrcoef(tr100_section)[np.triu_indices(n_regions)], 0, subject_id))                                 # compute the pearson corr. mat of the TR=100ms window, take triu entries.,  append  (triu ent, label, sub id)
            windows_sr1.append((np.corrcoef(tr100_section_noise)[np.triu_indices(n_regions)], 1, subject_id))                           # compute the pearson corr. mat of NOISY TR=100ms window, take triu entries., append (triu ent, label, sub id)


            windows_sr2.append((np.corrcoef(tr2150_section)[np.triu_indices(n_regions)], 0, subject_id))                                # compute the pearson corr. mat of the TR=2150ms window, take triu entries., append    (triu ent, label, sub id)
            windows_sr2.append((np.corrcoef(tr2150_section_noise)[np.triu_indices(n_regions)], 1, subject_id))                          # compute the pearson corr. mat of NOISY TR=2150ms window, take triu entries., append  (triu ent, label, sub id) 


            windows_concat.append((np.concatenate((windows_sr1[window_ix][0], windows_sr2[window_ix][0])),      # concat both triu entries
                                   0,                                                                           # get label of sr1&2 window
                                   subject_id))                                                                 # get subid of sr1&2 window
            
            windows_concat.append((np.concatenate((windows_sr1[window_ix+1][0], windows_sr2[window_ix+1][0])), # concat both noisy triu entries
                                   1,                                                                          # get label of sr1&2 window
                                   subject_id))                                                                # get subid of sr1&2 window
            
            windows_add.append((windows_sr1[window_ix][0] + windows_sr2[window_ix][0],                         # sum features of both triu entries
                                   0,                                                                          # get label of sr1&2 window
                                   subject_id))                                                                # get subid of sr1&2 window
                                             
            windows_add.append((windows_sr1[window_ix+1][0] + windows_sr2[window_ix+1][0],                     # sum features of noisy both triu entries
                                   1,                                                                          # get label of sr1&2 window
                                   subject_id))                                                                # get subid of sr1&2 window
                
            tr100_start_ix += tr100_stride                                                                                              # update sr1 start ix by adding stride length
            tr100_end_ix = tr100_end_ix + tr100_stride                                                                                  # update sr1 end ix by adding stride length to prev end ix
                
            tr2150_start_ix += tr2150_stride                                                                                            # update sr2 start ix by adding stride length
            tr2150_end_ix = tr2150_end_ix + tr2150_stride                                                                               # update sr2 end ix by adding stride length to prev end ix



    
    data_sr1 = [entry[0] for entry in windows_sr1]                                                                                   # collect datapoint from sr1 windows list entry tuple
    labels_sr1 = [entry[1] for entry in windows_sr1]                                                                                 # collect label from sr1 windows list entry tuple
    groups_sr1 = [entry[2] for entry in windows_sr1]                                                                                 # collect group from sr1 windows list entry tuple

    data_sr2 = [entry[0] for entry in windows_sr2]                                                                                   # collect datapoint from sr2 windows list entry tuple
    labels_sr2 = [entry[1] for entry in windows_sr2]                                                                                 # collect label from sr2 windows list entry tuple
    groups_sr2 = [entry[2] for entry in windows_sr2]                                                                                 # collect group from sr2 windows list entry tuple

    data_concat = [entry[0] for entry in windows_concat]                                                                             # collect datapoint from concat windows list entry tuple
    labels_concat = [entry[1] for entry in windows_concat]                                                                           # collect datapoint from concat windows list entry tuple
    groups_concat = [entry[2] for entry in windows_concat]                                                                           # collect group from concat windows list entry tuple

    data_add = [entry[0] for entry in windows_add]                                                                                   # collect datapoint from add windows list entry tuple
    labels_add = [entry[1] for entry in windows_add]                                                                                 # collect datapoint from add windows list entry tuple
    groups_add = [entry[2] for entry in windows_add]                                                                                 # collect group from add windows list entry tuple
    


    del tc_sr1                                                                                                                       # delete sr1 TC dict from memory since no longer used
    del tc_sr2                                                                                                                       # delete sr2 TC dict from memory since no longer used
    del tc_sr1_noise                                                                                                                 # delete noisy sr1 TC dict from memory since no longer used
    del tc_sr2_noise                                                                                                                 # delete noisy sr2 TC dict from memory since no longer used
    
    del windows_sr1                                                                                                                  # delete windows sr1 dict from memory since no longer used
    del windows_sr2                                                                                                                  # delete windows sr2 dict from memory since no longer used
    del windows_concat                                                                                                               # delete windows concat dict from memory since no longer used
    del windows_add                                                                                                                  # delete windows add dict from memory since no longer used

    
    


    # Initialize SimpleImputer
    imputer = SimpleImputer(strategy='mean')

    # Impute missing values and encode labels for each dataset
    data_sr1 = imputer.fit_transform(data_sr1)
    data_sr2 = imputer.fit_transform(data_sr2)
    data_concat = imputer.fit_transform(data_concat)
    data_add = imputer.fit_transform(data_add)

    np.random.seed(42)

    random_state = check_random_state(42)




    # run polyssifier for SR1 noise vs none
    logging.info('\n\n\n\n START POLYSSIFIER FOR TR=100ms')
    scaler1 = MinMaxScaler()#StandardScaler()
    data_scaled1 = scaler1.fit_transform(data_sr1)
    report1 = poly(data=data_scaled1, label=np.array(labels_sr1), groups=groups_sr1, n_folds=n_folds, random_state=random_state,                               # run polyssifier on SR1 noise vs none data, labels, and groups
                            project_name=f'SR1_noise_{scalar}', scale=False, concurrency=n_threads, save=False,
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    
    
    for classifier in report1.scores.columns.levels[0]:                                                                                         # iterate through all classifiers in the report
        if classifier == 'Voting':
            continue

        res1.append({'graph_no': graph_ix,                                                                                                      # save the SR1 results to a dict for results dataframe
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




    # run polyssifier for SR2 noise vs none
    logging.info('\n\n\n\n START POLYSSIFIER FOR TR=2150ms')
    scaler2 = MinMaxScaler()#StandardScaler()
    data_scaled2 = scaler2.fit_transform(data_sr2)
    report2 = poly(data=data_scaled2, label=np.array(labels_sr2), groups=groups_sr2, n_folds=n_folds, random_state=random_state,                              # run polyssifier on SR2 noise vs none data, labels, and groups
                            project_name=f'SR2_noise_{scalar}', scale=False, concurrency=n_threads, save=False,
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    
    
    for classifier in report2.scores.columns.levels[0]:                                                                                        # iterate through all classifiers in the report
        if classifier == 'Voting':
            continue

        res2.append({'graph_no': graph_ix,                                                                                                     # save the SR2 results to a dict for results dataframe
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





    # run polyssifier for CONCAT noise vs none
    logging.info('\n\n\n\n START POLYSSIFIER FOR CONCAT')
    scaler3 = MinMaxScaler()#StandardScaler()
    data_scaled3 = scaler3.fit_transform(data_concat)
    report3 = poly(data=data_scaled3, label=np.array(labels_concat), groups=groups_concat, n_folds=n_folds, random_state=random_state,                      # run polyssifier on CONCAT noise vs none data, labels, and groups
                            project_name=f'CONCAT_noise_{scalar}', scale=False, concurrency=n_threads, save=False,
                            exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    
    
    for classifier in report3.scores.columns.levels[0]:                                                                                      # iterate through all classifiers in the report
        if classifier == 'Voting':
            continue

        
        res3.append({'graph_no': graph_ix,                                                                                                   # save the CONCAT results to a dict for results dataframe
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





    # run polyssifier for ADD noise vs none
    logging.info('\n\n\n\n START POLYSSIFIER FOR ADD')
    scaler4 = MinMaxScaler()#StandardScaler()
    data_scaled4 = scaler4.fit_transform(data_add)
    report4 = poly(data=data_scaled4, label=np.array(labels_add), groups=groups_add, n_folds=n_folds, random_state=random_state,                          # run polyssifier on ADD noise vs none data, labels, and groups
                        project_name=f'ADD_noise_{scalar}', scale=False, concurrency=n_threads, save=False,
                        exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],  scoring='auc')
    

    for classifier in report4.scores.columns.levels[0]:                                                                                    # iterate through all classifiers in the report
        if classifier == 'Voting':
            continue

        
        res4.append({'graph_no': graph_ix,                                                                                                 # save the ADD results to a dict for results dataframe
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



df1 = pd.DataFrame(res1)                                                                                    # save SR1 results as dataframe
df2 = pd.DataFrame(res2)                                                                                    # save SR2 results as dataframe
df3 = pd.DataFrame(res3)                                                                                    # save CONCAT results as dataframe
df4 = pd.DataFrame(res4)                                                                                    # save ADD results as dataframe



df1.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr1_{SNR}_{graph_ix}.pkl')      # write SR1 dataframe to disk as pickle
df2.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr2_{SNR}_{graph_ix}.pkl')      # write SR2 dataframe to disk as pickle
df3.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/concat_{SNR}_{graph_ix}.pkl')   # write CONCAT dataframe to disk as pickle
df4.to_pickle(f'/data/users2/jwardell1/undersampling-project/OULU/pkl-files/add_{SNR}_{graph_ix}.pkl')      # write ADD dataframe to disk as pickle

