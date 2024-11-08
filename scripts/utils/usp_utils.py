from datetime import *
import logging
import pickle
from datetime import *
from pathlib import Path
import time
import os

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.signal import detrend

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


if os.getenv('THUNDERSVM_ACTIVE') == 'true':
    from thundersvm import SVC


def scale_noise(n, x, SNR):
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
    return colored_noise


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
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    X = data_df[f'{name}_Window']
    X = np.array([np.array(entry) for entry in X])
    return X, y, group


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
    cov_mat = data_params['cov_mat']

    if cov_mat:
        covariance_matrix = data_params['covariance_matrix']
    else:
        correlation_matrix = data_params['correlation_matrix']
    
    
    L = data_params['L']


    subjects = data_params['subjects']
    NOISE_SIZE = data_params['NOISE_SIZE']
    undersampling_rate = data_params['undersampling_rate']
    SNR = data_params['SNR']
    
    


    noises = {}
    ################ loading and preprocessing
    all_data = []
    for subject in subjects:
        if (noise_dataset == 'FBIRN') or (noise_dataset == 'COBRE'):
            noises[subject] = create_colored_noise(covariance_matrix, L, NOISE_SIZE) if cov_mat \
                  else create_colored_noise(correlation_matrix, L, NOISE_SIZE)
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


def get_resultpath(data_params):
    model_type = data_params['model_type']
    kernel_type = data_params['kernel_type']
    signal_dataset = data_params['signal_dataset']
    project_dir = data_params['project_dir']
    result_path = f'{project_dir}/assets/model_parameters/{signal_dataset}/{kernel_type}' if model_type == 'svm' \
                    else f'{project_dir}/assets/model_parameters/{signal_dataset}/{model_type}'
    return result_path


def get_filename(data_params):
    model_type = data_params['model_type']
    kernel_type = data_params['kernel_type']
    signal_dataset = data_params['signal_dataset']
    noise_dataset = data_params['noise_dataset']
    sampler = data_params['sampler']
    SNR = data_params['SNR']
    sr = data_params['sr']
    filename = f'{sr}_best_model_SNR_{SNR}_{kernel_type.upper()}_{signal_dataset}_{noise_dataset}_optuna_{sampler}.pkl' if model_type == 'svm' \
                    else f'{sr}_best_model_SNR_{SNR}_{model_type.upper()}_{signal_dataset}_{noise_dataset}_optuna_{sampler}.pkl'
    return filename



def write_results_to_pickle(data, data_params, key):
    model_type = data_params['model_type']
    kernel_type = data_params['kernel_type']
    signal_dataset = data_params['signal_dataset']
    noise_dataset = data_params['noise_dataset']
    SNR = data_params['SNR']
    pkl_dir = data_params['pkl_dir']
    if data != []:
        df = pd.DataFrame(data)
        current_date = datetime.now().strftime('%Y-%m-%d') + '-' + str(int(time.time()))
        month_date = '{}-{}'.format(datetime.now().strftime('%m'), datetime.now().strftime('%d'))
        
        filename = f'{key}_{SNR}_{noise_dataset}_{signal_dataset}{model_type}_{kernel_type}_{current_date}_optuna.pkl'
        
        directory = Path(f'{pkl_dir}/{month_date}')
        directory.mkdir(parents=True, exist_ok=True)

        df.to_pickle(f'{directory}/{filename}')
        logging.info(f'Saved results for {key} at {directory}/{filename}')



def set_data_params(args, project_dir):
    data_params = {}
    data_params['project_dir'] = project_dir


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
    
    
    
    log_level = 'DEBUG' if args.verbose else 'INFO'

    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    
    signal_dataset = args.signal_dataset.upper()    
    noise_dataset = args.noise_dataset.upper()




    if hasattr(args, 'subject_id'):
        subject_id = args.subject_id if args.subject_id != None else '000300655084'
    else:
        subject_id = '000300655084' if noise_dataset.lower() == 'fbirn' else '0'

    if noise_dataset.lower() == 'cobre':
        subject_id = int(subject_id)

    if hasattr(args, 'n_folds'):
        n_folds = args.n_folds if args.n_folds != None else 7
    else:
        n_folds = 7

    if hasattr(args, 'num_noise'):
        num_noise = args.num_noise if args.num_noise != None else 1
    else:
        num_noise = 1
    
    if hasattr(args, 'sampler'):
        sampler = args.sampler if args.sampler != None else 'tpe'
    else:
        sampler = 'tpe'

    if hasattr(args, 'kernel_type'):
        kernel_type = args.kernel_type if args.kernel_type != None else 'none'
    else:
        kernel_type = 'none'

    if hasattr(args, 'cov_mat'):
        cov_mat = args.cov_mat
    else:
        cov_mat = False


    signal_data = pd.read_pickle(f'{project_dir}/assets/data/{signal_dataset}_data.pkl')
    noise_data = pd.read_pickle(f'{project_dir}/assets/data/cov/{noise_dataset}_data.pkl') if cov_mat \
        else pd.read_pickle(f'{project_dir}/assets/data/{noise_dataset}_data.pkl')


    

    L, correlation_matrix = get_subject_data(subject_id, noise_data)

    data_params['correlation_matrix'] = correlation_matrix

    if cov_mat:
        covariance_matrix = noise_data['cov_mat']    #TODO load old dict
        logging.debug(f'covariance_matrix {covariance_matrix}')
        data_params['covariance_matrix'] = covariance_matrix

    logging.debug(f'L {L}')
    logging.debug(f'correlation_matrix {correlation_matrix}')

    data_params['L'] = L


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
    data_params["num_noise"] = num_noise
    data_params["kernel_type"] = kernel_type
    data_params['cov_mat'] = cov_mat
    data_params['subject_id'] = subject_id 


    return data_params




def set_hps_params(model_type, kernel_type, trial):
    hps_params = {}

    # Hyperparameter search space
    if model_type == 'svm':
        C = trial.suggest_float('C', 1, 1e3, log=True)
        gamma = trial.suggest_float('gamma', 1e-5, 1, log=True)
        tol = trial.suggest_float('tol', 1e-6, 2, log=True)
        
        hps_params['C'] = C
        hps_params['gamma'] = gamma
        hps_params['tol'] = tol 
        hps_params['kernel_type'] = tol 

    if model_type == 'lr':
        C = trial.suggest_float('C', 0.001, 1, log=True)

        hps_params['C'] = C
    
    if model_type == 'mlp':
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 100)])
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)

        hps_params['hidden_layer_sizes'] = hidden_layer_sizes
        hps_params['activation'] = activation
        hps_params['solver'] = solver
        hps_params['alpha'] = alpha
        hps_params['learning_rate'] = learning_rate
        hps_params['learning_rate_init'] = learning_rate_init

    if model_type == 'nb':
            hps_params['nb_type'] = trial.suggest_categorical("nb_type", ["gaussian", "multinomial"])

            if hps_params['nb_type'] == "multinomial":
                hps_params['alpha'] = trial.suggest_float("alpha", 1e-3, 10, log=True)
    
    hps_params['kernel_type'] = kernel_type

    return hps_params

def set_scaler(model_type, hps_params):
    if model_type != 'nb':
        scaler = StandardScaler()
    if model_type == 'nb':
        if hps_params['nb_type'] == "multinomial":
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()
    
    return scaler
    
def set_model(model_type, hps_params):
    if model_type == 'svm':
        model = SVC(kernel=hps_params['kernel_type'], C=hps_params['C'], gamma=hps_params['gamma'], tol=hps_params['tol'])
    
    if model_type == 'lr':
        model = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2', max_iter=150)

    if model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=hps_params['hidden_layer_sizes'],
            activation=hps_params['activation'],
            solver=hps_params['solver'],
            alpha=hps_params['alpha'],
            learning_rate=hps_params['learning_rate'],
            learning_rate_init=hps_params['learning_rate_init'],
            max_iter=200,
            random_state=42
        )
    
    if model_type == 'nb':
        if hps_params['nb_type'] == "gaussian":
            model = GaussianNB()

        if hps_params['nb_type'] == "multinomial":
            model = MultinomialNB(alpha=hps_params['alpha'])

    return model


def save_best_hyperparameters(data_params, best_trial):
    result_path = get_resultpath(data_params)
    filename = get_filename(data_params)
        
    
    result_file = f'{result_path}/{filename}'

    directory = Path(result_path)
    directory.mkdir(parents=True, exist_ok=True)

    with open(result_file, 'wb') as f:
        pickle.dump(best_trial.params, f)





    

def get_subject_data(subject_id, noise_data):
    sub_data = noise_data[noise_data['subject'] == subject_id]
    L = sub_data['L'].iloc[0]
    corr_mat = sub_data['corr_mat'].iloc[0]
    return L, corr_mat