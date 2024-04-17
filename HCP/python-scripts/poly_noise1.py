import numpy as np
import pandas as pd

from polyssifier import poly

import logging 

import scipy.io
from scipy.stats import zscore
import scipy.signal


from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.impute import SimpleImputer




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#Step 0: Iterate through values for nstd, burnin, noise_svar
SNRs = np.linspace(2, -2, 50) #[2, 1, 0.5, 0.1, 0, -0.1, -0.5, -1, -2]

nstd = 1.0
burn = 100

subjects = np.loadtxt("/data/users2/jwardell1/undersampling-project/HCP/txt-files/subjects.txt", dtype=str)

res1 = []
res2 = []
res3 = []
res4 = []


for SNR in SNRs:
    scalar = 10**(SNR/-2)
    logging.info(f'\t\t\t\tSNR- {SNR}')
    logging.info(f'\t\t\t\tscalar- {scalar}')
 

    #Step 1: Load data, compute noise, perform zscoring, add noise to loaded data
    num_converged = 0
    noises = dict()
    converged_subjects = []

    with open('/data/users2/jwardell1/undersampling-project/HCP/txt-files/sub_out_dirs.txt', 'r') as file:
        lines = file.readlines()

    
    for i in range(len(lines)):
        sub_out_dir = lines[i].strip()
        var_noise = np.load(f'{sub_out_dir}/var_noise.npy')

        
        # zscore var_noise
        var_noise = zscore(var_noise, axis=1)
        var_noise = var_noise * scalar

        noises[subjects[i]] = var_noise
        

    tc_sr1 = dict()
    tc_sr2 = dict()
    tc_sr1_noise = dict()
    tc_sr2_noise = dict()
    with open('/data/users2/jwardell1/undersampling-project/HCP/txt-files/tc_data.txt', 'r') as tc_data:
        lines = tc_data.readlines()
    
    fnc_sr1 = []
    fnc_sr2 = []
    fnc_concat = []
    fnc_add = []

    for i in range(len(lines)):
        file_path_sr1 = lines[i].strip()
        logging.info(f'Processing SR1: {file_path_sr1}')
        try:
            sr1 = scipy.io.loadmat(file_path_sr1)['TCMax']
        except:
            continue

        if sr1.shape[0] != 53:
            sr1 = sr1.T

        if sr1.shape[1] < 1200:
            continue

        logging.info(f'sr1.shape - {sr1.shape}')#convert to logger
        sr1 = zscore(sr1, axis=1)
        sr1 = scipy.signal.detrend(sr1, axis=1)
        sr2 = sr1[:,::3]

        n_regions = sr1.shape[0]


        fnc_sr1.insert(i, (np.corrcoef(sr1)[np.triu_indices(n_regions)], 0, subjects[i]))
        fnc_sr1.insert(i+1, (np.corrcoef(sr1 + var_noise)[np.triu_indices(n_regions)], 1, subjects[i]))

        fnc_sr2.insert(i, (np.corrcoef(sr2)[np.triu_indices(n_regions)], 0, subjects[i]))
        fnc_sr2.insert(i+1, (np.corrcoef(sr2 + var_noise[:,::3])[np.triu_indices(n_regions)], 1, subjects[i]))

        fnc_concat.insert(i, (np.concatenate((fnc_sr1[i][0], fnc_sr2[i][0])), 0, subjects[i]))
        fnc_concat.insert(i, (np.concatenate((fnc_sr1[i+1][0], fnc_sr2[i+1][0])), 1, subjects[i]))

        fnc_add.insert(i, ((fnc_sr1[i][0] + fnc_sr2[i][0]), 0, subjects[i]))
        fnc_add.insert(i, ((fnc_sr1[i+1][0] + fnc_sr2[i+1][0]), 1, subjects[i]))

    data_sr1 = [entry[0] for entry in fnc_sr1]
    labels_sr1 = [entry[1] for entry in fnc_sr1]
    groups_sr1 = [entry[2] for entry in fnc_sr1]

    data_sr2 = [entry[0] for entry in fnc_sr2]
    labels_sr2 = [entry[1] for entry in fnc_sr2]
    groups_sr2 = [entry[2] for entry in fnc_sr2]

    data_concat = [entry[0] for entry in fnc_concat]
    labels_concat = [entry[1] for entry in fnc_concat]
    groups_concat = [entry[2] for entry in fnc_concat]

    data_add = [entry[0] for entry in fnc_add]
    labels_add = [entry[1] for entry in fnc_add]
    groups_add = [entry[2] for entry in fnc_add]
    
    
    
    # Initialize SimpleImputer
    imputer = SimpleImputer(strategy='mean')

    # Impute missing values and encode labels for each dataset
    data_sr1 = imputer.fit_transform(data_sr1)
    data_sr2 = imputer.fit_transform(data_sr2)
    data_concat = imputer.fit_transform(data_concat)
    data_add = imputer.fit_transform(data_add)


    np.random.seed(42)
    random_state = check_random_state(42)

    scaler1 = MinMaxScaler()#StandardScaler()
    data_scaled1 = scaler1.fit_transform(data_sr1)
    # Perform poly_subject and plot_scores for each datase
    report1 = poly(data_scaled1, np.array(labels_sr1), n_folds=5, scale=True, random_state=random_state,
                        exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],
                        scoring='f1', project_name=f'SR1_noise_{scalar}', concurrency=4, verbose=True)
    
    for classifier in report1.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res1.append({'nstd': nstd,
                        'burnin': burn,
                        'snr': SNR,
                        'scalar': scalar,
                        'classifier': classifier,
                        'test_scores': report1.scores[classifier, 'test'], 
                        'target': report1.target, 
                        'predictions': np.array(report1.predictions[classifier]).astype(int),
                        'test_proba': report1.test_proba[classifier]})



    scaler2 = MinMaxScaler()#StandardScaler()
    data_scaled2 = scaler2.fit_transform(data_sr2)
    report2 = poly(data_scaled2, np.array(labels_sr2), n_folds=5, scale=True, random_state=random_state,
                        exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'], 
                        scoring='f1', project_name=f'SR2_noise_{scalar}', concurrency=4, verbose=True)
    
    for classifier in report2.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res2.append({'nstd': nstd,
                    'burnin': burn,
                    'snr': SNR,
                    'scalar': scalar,
                    'classifier': classifier,
                    'test_scores': report2.scores[classifier, 'test'], 
                    'target': report2.target, 
                    'predictions': np.array(report2.predictions[classifier]).astype(int), 
                    'test_proba': report2.test_proba[classifier]})

    scaler3 = MinMaxScaler()#StandardScaler()
    data_scaled3 = scaler3.fit_transform(data_concat)
    report3 = poly(data_scaled3, np.array(labels_concat), n_folds=5, scale=True, random_state=random_state,
                        exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],
                        scoring='f1', project_name=f'CONCAT_noise_{scalar}', concurrency=4, verbose=True)
    for classifier in report3.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res3.append({'nstd': nstd,
                    'burnin': burn,
                    'snr': SNR,
                    'scalar': scalar,
                    'classifier': classifier,
                    'test_scores': report3.scores[classifier, 'test'], 
                    'target': report3.target, 
                    'predictions': np.array(report3.predictions[classifier]).astype(int), 
                    'test_proba': report3.test_proba[classifier]})
        
    scaler4 = MinMaxScaler()#StandardScaler()
    data_scaled4 = scaler4.fit_transform(data_add)
    report4 = poly(data_scaled4, np.array(labels_add), n_folds=5, scale=True, random_state=random_state,
                        exclude=['Decision Tree', 'Random Forest', 'Voting', 'Nearest Neighbors', 'Linear SVM'],
                        scoring='f1', project_name=f'CONCAT_noise_{scalar}', concurrency=4, verbose=True)
    
    for classifier in report4.scores.columns.levels[0]:
        if classifier == 'Voting':
            continue

        # Append the results to the list as a dictionary
        res4.append({'nstd': nstd,
                    'burnin': burn,
                    'snr': SNR,
                    'scalar': scalar,
                    'classifier': classifier,
                    'test_scores': report4.scores[classifier, 'test'], 
                    'target': report4.target, 
                    'predictions': np.array(report4.predictions[classifier]).astype(int), 
                    'test_proba': report4.test_proba[classifier]})



#populate dataframe here for combimation of noise
df1 = pd.DataFrame(res1)
df2 = pd.DataFrame(res2)
df3 = pd.DataFrame(res3)
df4 = pd.DataFrame(res4)
df1.to_pickle('/data/users2/jwardell1/undersampling-project/HCP/pkl-files/sr1.pkl')
df2.to_pickle('/data/users2/jwardell1/undersampling-project/HCP/pkl-files/sr2.pkl')
df3.to_pickle('/data/users2/jwardell1/undersampling-project/HCP/pkl-files/concat.pkl')
df4.to_pickle('/data/users2/jwardell1/undersampling-project/HCP/pkl-files/add.pkl')





    


