import numpy as np
import pandas as pd
from polyssifier import poly_subject
import os
import logging  # Import the logging module
import scipy.sparse as sp
import scipy.io
from scipy.sparse.linalg import eigs
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import graph2adj
import uuid  # Make sure to import the uuid module
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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


def drawsamplesLG(A, nstd=0.1, samples=10):
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data


def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1, dist="normal", nstd=0.1):
    Agt = A
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=nstd)
    data = data[:, burnin:]
    return data[:, ::rate]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#Step 0: Iterate through values for n_std, burnin, noise_svar
n_std = [1e-8, 1e-5, 1e-1]
burnin = [50, 100, 175]
noise_svar = [1e-8, 1e-5, 1e-1]
scalars = [1, 100]#, 50, 100]
NOISE_SIZE = 2961*2
NUM_SUBS = 10
subjects = [20150210, 20150417, 20150428, 20151110, 20151127, 
            20150410, 20150421, 20151030, 20151117, 20151204]

res1 = []
res2 = []
res3 = []

for nstd in n_std:
    for burn in burnin:
        for noise in noise_svar:
            for scalar in scalars:
                u_rate = 1
                g = gk.ringmore(53, 10)
                A = graph2adj(g)

                #Step 1: Load data, compute noise, perform zscoring, add noise to loaded data
                num_converged = 0
                noises = dict()
                converged_subjects = []

                with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/sub_out_dirs.txt', 'r') as file:
                    lines = file.readlines()

                while num_converged < len(lines):
                    for i in range(len(lines)):
                        if i in converged_subjects:
                            continue  
                        
                        sub_out_dir = lines[i].strip()
                        try:
                            W = create_stable_weighted_matrix(A, threshold=0.001, powers=[2])
                            var_noise = genData(W, rate=u_rate, burnin=burn, ssize=NOISE_SIZE, noise=noise, nstd=nstd)
                        except Exception as e:
                            print(f'Convergence error while generating matrix for dir {sub_out_dir}, num converged: {num_converged}')
                            print(e)
                            continue

                        # zscore var_noise
                        mean = np.mean(var_noise, axis=1, keepdims=True)
                        std = np.std(var_noise, axis=1, keepdims=True)
                        var_noise = (var_noise - mean) / std
                        var_noise = var_noise / scalar

                        noises[subjects[i]] = var_noise
                        converged_subjects.append(i)
                        num_converged += 1


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

                    logging.info(f'sr1.shape - {sr1.shape}')#convert to logger
                    logging.info(f'sr2.shape - {sr2.shape}')#convert to logger

                    #zscore tc_data
                    mean = np.mean(sr1, axis=1, keepdims=True)
                    std = np.std(sr1, axis=1, keepdims=True)
                    sr1 = (sr1 - mean) / std

                    mean = np.mean(sr2, axis=1, keepdims=True)
                    std = np.std(sr2, axis=1, keepdims=True)
                    sr2 = (sr2 - mean) / std

                    tc_sr1[subjects[i//2]] = sr1 #TR=100ms
                    tc_sr2[subjects[i//2]] = sr2 #TR=2150ms

                    tc_sr1_noise[subjects[i//2]] = sr1 + var_noise[:,::2]
                    tc_sr2_noise[subjects[i//2]] = sr2 + var_noise[:,::33]

                #TODO- Step 2: Perform windowing on noise/non-noise data
                windows_sr1 = []
                windows_sr2 = []
                windows_sr1_noise = []
                windows_sr2_noise = []
                windows_concat = []
                windows_concat_noise = []
                lsr1 = []
                lsr1n = []
                lsr2 = []
                lsr2n = []
                lconcat = []
                lconcatn = []
                g1 = []
                g1n = []
                g2 = []
                g2n = []
                gc = []
                gcn = []
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


                        windows_sr1.append(np.corrcoef(tr100_section)[np.triu_indices(n_regions)])
                        windows_sr1_noise.append(np.corrcoef(tr100_section_noise)[np.triu_indices(n_regions)])
                        lsr1.append('0') #no-noise: class label 0
                        lsr1n.append('1') #noise-present: class label 1
                        g1.append(subject_id)
                        g1n.append(subject_id)


                        windows_sr2.append(np.corrcoef(tr2150_section)[np.triu_indices(n_regions)])
                        windows_sr2_noise.append(np.corrcoef(tr2150_section_noise)[np.triu_indices(n_regions)])
                        lsr2.append('0') #no-noise: class label 0
                        lsr2n.append('1') #noise-present: class label 1
                        g2.append(subject_id)
                        g2n.append(subject_id)


                        windows_concat.append(np.concatenate((windows_sr1[j], windows_sr2[j])))
                        windows_concat_noise.append(np.concatenate((windows_sr1_noise[j], windows_sr2_noise[j])))
                        lconcat.append('0') #no-noise: class label 0
                        lconcatn.append('1') #noise-present: class label 1
                        gc.append(subject_id)
                        gcn.append(subject_id)

                        tr100_start_ix += tr100_stride
                        tr100_end_ix = tr100_end_ix + tr100_stride
                            
                        tr2150_start_ix += tr2150_stride
                        tr2150_end_ix = tr2150_end_ix + tr2150_stride

                del tc_sr1
                del tc_sr2
                del tc_sr1_noise
                del tc_sr2_noise


                #TODO- Step 3: Run polyssifier on windowed data
                data_sr1 = np.concatenate((windows_sr1, windows_sr1_noise))
                labels_sr1 = np.concatenate((lsr1, lsr1n))
                groups_sr1 = np.concatenate((g1, g1n))

                data_sr2 = np.concatenate((windows_sr2, windows_sr2_noise))
                labels_sr2 = np.concatenate((lsr2, lsr2n))
                groups_sr2 = np.concatenate((g2, g2n))

                data_concat = np.concatenate((windows_concat, windows_concat_noise))
                labels_concat = np.concatenate((lconcat, lconcatn))
                groups_concat = np.concatenate((gc, gcn))
                from sklearn.impute import SimpleImputer

                from sklearn.preprocessing import LabelEncoder


                # Initialize SimpleImputer
                imputer = SimpleImputer(strategy='mean')

                # Impute missing values and encode labels for each dataset
                data_sr1 = imputer.fit_transform(data_sr1)
                data_sr2 = imputer.fit_transform(data_sr2)
                data_concat = imputer.fit_transform(data_concat)

                # Initialize LabelEncoder
                label_encoder = LabelEncoder()

                # Encode string labels into numeric values
                labels_sr1_encoded = label_encoder.fit_transform(labels_sr1)
                labels_sr2_encoded = label_encoder.fit_transform(labels_sr1)
                labels_concat_encoded = label_encoder.fit_transform(labels_concat)


                noise_dir = '/data/users2/jwardell1/undersampling-project/OULU/images/noise_comparison'
                #visualize a random datapoint of each class from sr1, sr2, and concat
                random_data_points_labels = []
                random_data_points = []
                classes = [0,1]
                for data, labels in [(data_sr1, labels_sr1_encoded), (data_sr2, labels_sr2_encoded), (data_concat, labels_concat_encoded)]:
                    for class_label in classes:
                        indices = np.where(labels == class_label)[0]
                        random_index = np.random.choice(indices)
                        random_data_point = data[random_index]
                        random_data_points.append(random_data_point)
                        random_data_points_labels.append(class_label)

                for ix in range(len(random_data_points)):
                    plt.clf()
                    n = 53
                    triu_indices = np.triu_indices(n)
                    if len(random_data_points[ix]) >1431:
                        fnc1 = random_data_points[ix][:1431]
                        mat1 = np.zeros((n,n))
                        mat1[triu_indices] = fnc1
                        mat1 = mat1 + mat1.T - np.diag(np.diag(mat1))
                        plt.imshow(mat1)
                        if random_data_points_labels[ix] == 0:
                            plt.title(f'Concat1- Label:{random_data_points_labels[ix]}')
                            plt.savefig(f'{noise_dir}/{nstd}_{burn}_{noise}_{scalar}_sample:{ix}.1_label:{random_data_points_labels[ix]}.png')
                        else:
                            plt.title(f'Concat1- Label:{labels_sr1[len(labels_sr1)-1]}    n_std:{nstd}    burnin:{burn}     noise_svar:{noise}     scalar:{scalar}')
                            plt.savefig(f'{noise_dir}/{nstd}_{burn}_{noise}_{scalar}_sample:{ix}.1_label:{random_data_points_labels[ix]}.png')
                        plt.clf()
                        fnc2 = random_data_points[ix][1431:]
                        mat2 = np.zeros((n,n))
                        mat2[triu_indices] = fnc2
                        mat2 = mat2 + mat2.T - np.diag(np.diag(mat2))
                        plt.imshow(mat2)
                        if random_data_points_labels[ix] == 0:
                            plt.title(f'Concat2- Label:{random_data_points_labels[ix]}')
                            plt.savefig(f'{noise_dir}/{nstd}_{burn}_{noise}_{scalar}_sample:{ix}.2_label:{random_data_points_labels[ix]}.png')
                        else:
                            plt.title(f'Concat2- Label:{labels_sr1[len(labels_sr1)-1]}    n_std:{nstd}    burnin:{burn}     noise_svar:{noise}     scalar:{scalar}')
                            plt.savefig(f'{noise_dir}/{nstd}_{burn}_{noise}_{scalar}_sample:{ix}.2_label:{random_data_points_labels[ix]}.png')

                        continue

                    mat = np.zeros((n,n))
                    mat[triu_indices] = random_data_points[ix]
                    mat = mat + mat.T - np.diag(np.diag(mat))
                    plt.imshow(mat)
                    if random_data_points_labels[ix] == 0:
                        plt.title(f'Sample:{ix}- Label:{random_data_points_labels[ix]}')
                    else:
                        plt.title(f'Sample:{ix}- Label:{random_data_points_labels[ix]}    n_std:{nstd}    burnin:{burn}     noise_svar:{noise}     scalar:{scalar}')
                
                    plt.savefig(f'{noise_dir}/{nstd}_{burn}_{noise}_{scalar}_sample:{ix}_label:{random_data_points_labels[ix]}.png')


                scaler1 = StandardScaler()
                data_scaled1 = scaler1.fit_transform(data_sr1)
                # Perform poly_subject and plot_scores for each dataset
                report1 = poly_subject(data_scaled1, labels_sr1_encoded, groups_sr1, n_folds=10, project_name='SR1', scale=True, exclude=['Decision Tree', 'Random Forest', 'Voting'],  scoring='f1')
                for classifier in report1.scores.columns.levels[0]:
                    if classifier == 'Voting':
                        continue

                    # Append the results to the list as a dictionary
                    res1.append({'n_std': nstd,
                                    'burnin': burn,
                                    'noise_svar': noise,
                                    'scalar': scalar,
                                    'classifier': classifier,
                                    'test_scores': report1.scores[classifier, 'test'], 
                                    'target': report1.target, 
                                    'predictions': np.array(report1.predictions[classifier]).astype(int),
                                    'test_proba': report1.test_proba[classifier]})



                scaler2 = StandardScaler()
                data_scaled2 = scaler2.fit_transform(data_sr2)
                report2 = poly_subject(data_scaled2, labels_sr2_encoded, groups_sr2, n_folds=10, project_name='SR2', scale=True, exclude=['Decision Tree', 'Random Forest', 'Voting'],  scoring='f1')
                for classifier in report2.scores.columns.levels[0]:
                    if classifier == 'Voting':
                        continue

                    # Append the results to the list as a dictionary
                    res2.append({'n_std': nstd,
                                'burnin': burn,
                                'noise_svar': noise,
                                'scalar': scalar,
                                'classifier': classifier,
                                'test_scores': report2.scores[classifier, 'test'], 
                                'target': report2.target, 
                                'predictions': np.array(report2.predictions[classifier]).astype(int), 
                                'test_proba': report2.test_proba[classifier]})

                scaler3 = StandardScaler()
                data_scaled3 = scaler3.fit_transform(data_concat)
                report3 = poly_subject(data_scaled3, labels_concat_encoded, groups_concat, n_folds=10, project_name='CONCAT', scale=True, exclude=['Decision Tree', 'Random Forest', 'Voting'],  scoring='f1')
                for classifier in report3.scores.columns.levels[0]:
                    if classifier == 'Voting':
                        continue

                    # Append the results to the list as a dictionary
                    res3.append({'n_std': nstd,
                                'burnin': burn,
                                'noise_svar': noise,
                                'scalar': scalar,
                                'classifier': classifier,
                                'test_scores': report3.scores[classifier, 'test'], 
                                'target': report3.target, 
                                'predictions': np.array(report3.predictions[classifier]).astype(int), 
                                'test_proba': report3.test_proba[classifier]})



#populate dataframe here for combimation of 'n_std', 'burnin', 'noise_svar
df1 = pd.DataFrame(res1)
df2 = pd.DataFrame(res2)
df3 = pd.DataFrame(res3)
df1.to_pickle('/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr1.pkl')
df2.to_pickle('/data/users2/jwardell1/undersampling-project/OULU/pkl-files/sr2.pkl')
df3.to_pickle('/data/users2/jwardell1/undersampling-project/OULU/pkl-files/concat.pkl')


'''
import seaborn as sns
import matplotlib.pyplot as plt

# Plot test score vs n_std for df1
plt.figure(figsize=(10, 6))
sns.lineplot(data=df1, x='n_std', y='avg_test_score', marker='o')
plt.title('Test Score vs n_std for df1')
plt.xlabel('n_std')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_n_std_df1.png')
plt.close()

# Plot test score vs burnin for df1
plt.figure(figsize=(10, 6))
sns.lineplot(data=df1, x='burnin', y='avg_test_score', marker='o')
plt.title('Test Score vs burnin for df1')
plt.xlabel('burnin')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_burnin_df1.png')
plt.close()

# Plot test score vs noise_svar for df1
plt.figure(figsize=(10, 6))
sns.lineplot(data=df1, x='noise_svar', y='avg_test_score', marker='o')
plt.title('Test Score vs noise_svar for df1')
plt.xlabel('noise_svar')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_noise_svar_df1.png')
plt.close()

# Plot test score vs n_std for df2
plt.figure(figsize=(10, 6))
sns.lineplot(data=df2, x='n_std', y='avg_test_score', marker='o')
plt.title('Test Score vs n_std for df2')
plt.xlabel('n_std')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_n_std_df2.png')
plt.close()

# Plot test score vs burnin for df2
plt.figure(figsize=(10, 6))
sns.lineplot(data=df2, x='burnin', y='avg_test_score', marker='o')
plt.title('Test Score vs burnin for df2')
plt.xlabel('burnin')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_burnin_df2.png')
plt.close()

# Plot test score vs noise_svar for df2
plt.figure(figsize=(10, 6))
sns.lineplot(data=df2, x='noise_svar', y='avg_test_score', marker='o')
plt.title('Test Score vs noise_svar for df2')
plt.xlabel('noise_svar')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_noise_svar_df2.png')
plt.close()

# Plot test score vs n_std for df3
plt.figure(figsize=(10, 6))
sns.lineplot(data=df3, x='n_std', y='avg_test_score', marker='o')
plt.title('Test Score vs n_std for df3')
plt.xlabel('n_std')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_n_std_df3.png')
plt.close()

# Plot test score vs burnin for df3
plt.figure(figsize=(10, 6))
sns.lineplot(data=df3, x='burnin', y='avg_test_score', marker='o')
plt.title('Test Score vs burnin for df3')
plt.xlabel('burnin')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_burnin_df3.png')
plt.close()

# Plot test score vs noise_svar for df3
plt.figure(figsize=(10, 6))
sns.lineplot(data=df3, x='noise_svar', y='avg_test_score', marker='o')
plt.title('Test Score vs noise_svar for df3')
plt.xlabel('noise_svar')
plt.ylabel('Average Test Score')
plt.savefig('test_score_vs_noise_svar_df3.png')
plt.close()
'''