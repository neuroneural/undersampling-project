import logging
import pickle

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.signal import detrend
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA



def scale_noise(n, x, SNR):
    #assert x.shape[0] == 53, 'timecourse dimension 0 should be 53'
    #assert n.shape[0] == 53, 'noise dimension 0 should be 53'
    xTx = np.sum(np.square(x))
    nTn = np.sum(np.square(n))
    if nTn == 0:
        return np.zeros_like(n)
    c = ((xTx / nTn)**0.5) / (10**(SNR/2)) 
    scaled_noise = c * n
    return scaled_noise


def create_colored_noise(corr_mat, L, noise_size):
    assert corr_mat.shape == (53, 53), 'cov_mat should be 53 x 53 matrix'
    assert L.shape == (53, 53), 'L should be 53 x 53 matrix'
    mean = np.zeros(corr_mat.shape[0])
    white_noise = np.random.multivariate_normal(mean, np.eye(corr_mat.shape[0]), size=noise_size)
    colored_noise = white_noise @ L.T
    colored_noise = colored_noise.T
    #colored_noise = zscore(colored_noise, axis=1)
    #colored_noise = detrend(colored_noise, axis=1)
    #the noise should have mean zero and covariance should not be the identity right ???
    return colored_noise


def create_var_noise(A, subjects, threshold, u_rate, burn, NOISE_SIZE, nstd):
    num_converged = 0
    converged_subjects = []
    noises = {}
    NUM_SUBS = len(subjects)
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
                logging.info(f'num converged: {num_converged}/{NUM_SUBS}')

            except Exception as e:
                print(e)
    return noises


def preprocess_timecourse(tc_data):
    #assert tc_data.shape[0] == 53, 'timecourse dimension 0 should be 53'
    data = detrend(tc_data, axis=1)   
    data = zscore(data, axis=1)
    return data
    

def parse_X_y_groups(data_df, name):
    le = LabelEncoder()
    group = le.fit_transform(data_df['subject'])
    y = data_df['target']
    y = np.array([str(entry) for entry in y])
    X = data_df[f'{name}_Window']
    X = np.array([np.array(entry) for entry in X])
    return X, y, group


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


def perform_windowing(data_df):
    sr1_data = []
    sr2_data = []
    add_data = []
    concat_data = []
    subjects = np.unique(data_df['Subject_ID'])
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
    
    return sr1_data, sr2_data, add_data, concat_data



def load_timecourses(signal_data, data_params):
    signal_dataset = data_params['signal_dataset']
    noise_dataset = data_params['noise_dataset']
    
    if noise_dataset == 'VAR':
        A = data_params['A']
        nstd = data_params['nstd']
        threshold = data_params['threshold']
        u_rate = data_params['u_rate']
        burn = data_params['burn']
    else:
        correlation_matrix = data_params['correlation_matrix']
        L = data_params['L']


    subjects = data_params['subjects']
    NOISE_SIZE = data_params['NOISE_SIZE']
    undersampling_rate = data_params['undersampling_rate']
    SNR = data_params['SNR']
    
    


    noises = {} if noise_dataset != 'VAR' else create_var_noise(A, subjects, threshold, u_rate, burn, NOISE_SIZE, nstd)
    ################ loading and preprocessing
    all_data = []
    for subject in subjects:
        if noise_dataset != 'VAR':
            noises[subject] = create_colored_noise(correlation_matrix, L, NOISE_SIZE)
            logging.debug(f'computed noise for subject: {subject}')

            if signal_dataset == 'SIMULATION': 
                noises[subject] = noises[subject][:5, :]
            logging.debug(f'noises[subject].shape {noises[subject].shape}')
                


        logging.debug(f'loading timecourse for subject {subject}')
        if signal_dataset == 'HCP': 
            logging.debug('HCP dataset detected during loading')
            sr1_tc = signal_data[
                signal_data['subject'] == subject
            ]['ica_timecourse'].iloc[0]
        

        else:
            tr1 = signal_data.iloc[0]['sampling_rate'].replace('TR', '')
            logging.debug(f'TR {tr1} detected during loading')
            sr1_tc = signal_data[
                (signal_data['subject'] == subject) & 
                (signal_data['sampling_rate'] == f'TR{tr1}')
            ]['ica_timecourse'].iloc[0]

            tr2 = signal_data.iloc[1]['sampling_rate'].replace('TR', '')
            logging.debug(f'TR {tr2} detected during loading')
            sr2_tc = signal_data[
                (signal_data['subject'] == subject) & 
                (signal_data['sampling_rate'] == f'TR{tr2}')
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
            k1 = 2 if signal_dataset == 'OULU' else NOISE_SIZE // sr1_tc.shape[1]
            k2 = 33 if signal_dataset == 'OULU' else NOISE_SIZE // sr2_tc.shape[1]

            noise_sr1 = scale_noise(noises[subject][:,::k1], sr1_tc, SNR)
            noise_sr2 = scale_noise(noises[subject][:,::k2], sr2_tc, SNR)




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
    
    return all_data


def plot_cv_indices(cv, X, y, group, ax, n_splits, save_data, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    group = np.array(group, dtype=int)
    y = np.array(y, dtype=int)
    print(set(group))
    cmap_cv = plt.cm.coolwarm
    cmap_data = plt.cm.Paired

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 1600],
    )
    fig = ax.get_figure()
    sampling_rate = save_data['sampling_rate']
    snr = save_data['snr']
    noise_dataset = save_data['noise_dataset']
    signal_dataset = save_data['signal_dataset']

    name = f'{sampling_rate}_{snr}_{noise_dataset}_{signal_dataset}'

    ax.set_title("{}_{}".format(type(cv).__name__, name), fontsize=15)


    fig.savefig(f'cvplot_{name}.png')


def set_data_params(args, project_dir):
    data_params = {}


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
    
    
    

    n_folds = args.n_folds if args.n_folds != None else 7
    log_level = 'DEBUG' if args.verbose else 'INFO'
    signal_dataset = args.signal_dataset.upper()
    noise_dataset = args.noise_dataset.upper()
    num_noise = args.num_noise if args.num_noise != None else 1
    sampler = args.sampler if args.sampler != None else 'tpe'
    model_type = args.model_type

    signal_data = pd.read_pickle(f'{project_dir}/assets/data/{signal_dataset}_data.pkl')
    noise_data = scipy.io.loadmat(f'{project_dir}/assets/data/{noise_dataset}_data.mat')

    if noise_dataset == "VAR":
        A = noise_data['A']
        u_rate = 1
        nstd = 1.0
        burn = 100
        threshold = 0.0001
        
        logging.debug(f'A - {A}')
        logging.debug(f'u_rate - {u_rate}')
        logging.debug(f'nstd - {nstd}')
        logging.debug(f'burn - {burn}')
        logging.debug(f'threshold - {threshold}')


        data_params['A'] = A
        data_params['u_rate'] = u_rate
        data_params['nstd'] = nstd
        data_params['burn'] = burn
        data_params['threshold'] = threshold

    else:
        L = noise_data['L']
        correlation_matrix = noise_data['corr_mat']

        logging.debug(f'L {L}')
        logging.debug(f'correlation_matrix {correlation_matrix}')

        data_params['L'] = L
        data_params['correlation_matrix'] = correlation_matrix


    if signal_dataset == 'OULU':
        undersampling_rate = 1
        NOISE_SIZE = 2961*2
    
    if signal_dataset == 'SIMULATION':
        undersampling_rate = 1
        NOISE_SIZE = 18018 #might should write a function to compute this, it is LCM(t1*k1, t2*k2)

    if signal_dataset == 'HCP':
        NOISE_SIZE = 1200
        undersampling_rate = 6

    subjects = np.unique(signal_data['subject'])
    
    data_params['subjects'] = subjects
    data_params['noise_dataset'] = noise_dataset
    data_params['signal_dataset'] = signal_dataset
    data_params['SNRs'] = SNRs
    data_params['n_folds'] = n_folds
    data_params['log_level'] = log_level
    data_params['signal_dataset'] = signal_dataset
    data_params['noise_dataset'] = noise_dataset
    data_params['sampler'] = sampler
    data_params['signal_data'] = signal_data
    data_params['noise_data'] = noise_data
    data_params['undersampling_rate'] = undersampling_rate
    data_params['NOISE_SIZE'] = NOISE_SIZE
    data_params['model_type'] = model_type
    data_params["num_noise"] = num_noise



    return data_params