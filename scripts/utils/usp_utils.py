import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.signal import detrend
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder



def scale_noise(n, x, SNR):
    assert x.shape[0] == 53, 'timecourse dimension 0 should be 53'
    assert n.shape[0] == 53, 'noise dimension 0 should be 53'
    xTx = np.sum(np.square(x))
    nTn = np.sum(np.square(n))
    if nTn == 0:
        return np.zeros_like(n)
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
    assert tc_data.shape[0] == 53, 'timecourse dimension 0 should be 53'
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


def tune_svm(X, y, group, param_grid):
    k_values = param_grid['k_values']
    name = param_grid['name']

    param_grid = {
        'C' : param_grid['C'],
        'tol' : param_grid['tol'],
    }


    best_auc = -float('inf')
    best_model = None
    best_params = None

    # Iterate over the different k values for StratifiedGroupKFold
    for k in k_values:
        logging.info(f"Testing with k={k} folds in StratifiedGroupKFold")

        # Outer cross-validation loop
        outer_skgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        outer_cv_results = []

        for train_idx, test_idx in outer_skgkf.split(X, y, groups=group):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            group_train, group_test = group[train_idx], group[test_idx]

            # Inner cross-validation with GridSearchCV
            inner_skgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
            svm = SVC(kernel='rbf', probability=True)
            
            # Define the scoring strategy
            scoring = make_scorer(roc_auc_score, needs_proba=True)
            
            grid_search = GridSearchCV(svm, param_grid, scoring=scoring, cv=inner_skgkf, n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train, groups=group_train)
            
            # Evaluate the model on the outer fold's test set
            best_model_fold = grid_search.best_estimator_
            y_pred_proba = best_model_fold.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            outer_cv_results.append(auc_score)

            # Track the best model based on AUC
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = best_model_fold
                best_params = grid_search.best_params_

        # Report results
        logging.info(f"Best model for {name} with k={k}: C={best_params['C']}, tol={best_params['tol']}, ROC AUC={best_auc:.4f}")
        logging.info(f"Mean AUC across outer folds for k={k}: {np.mean(outer_cv_results):.4f}")


    return best_model


def fit_svm(X, y, group, model_info, k):
    sampling_rates = ['sr1', 'sr2', 'add', 'concat']

    weights_dir = model_info['weights_dir']
    model_filename = model_info['model_filename']
    model_path = f'{weights_dir}/{model_filename}'
    
    filename_str = model_filename.split('_')
    sampling_rate = filename_str[0] if filename_str[0] in sampling_rates else filename_str[1]
    snr = filename_str[4] if filename_str[4] != 'SNR' else filename_str[5]
    logging.info(f'sampling_rate {sampling_rate}')
    logging.info(f'snr {snr}')


    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    fold_scores = {}
    for fold_ix, (train_index, test_index) in enumerate(sgkf.split(X, y, group), start=0):
        _, X_test = X[train_index], X[test_index]
        _, y_test = y[train_index], y[test_index]

        with open(model_path, 'rb') as file:
            svm = pickle.load(file)

        y_pred = svm.predict(X_test).astype(np.int8)
        y_test = y_test.astype(np.int8)
        
        fold_score = roc_auc_score(y_test, y_pred)
        fold_scores[fold_ix] = fold_score
        test_subs = set(group[test_index])
        plot_and_save_confusion_matrix(y_test, y_pred, sampling_rate, snr, fold_ix, test_subs)

        
    return fold_scores



def plot_and_save_confusion_matrix(y_true, y_pred, save_data):
    sampling_rate = save_data['sampling_rate']
    snr = save_data['snr']
    fold_ix = save_data['fold_ix']
    test_subs = save_data['test_subs']
    noise_dataset = save_data['noise_dataset']
    signal_dataset = save_data['signal_dataset']
    noise_ix = save_data['noise_ix']

    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    auc = roc_auc_score(y_true, y_pred)
    
    # Create the plot title
    auc_rounded = f'{auc:.3g}'
    plot_title = f'{sampling_rate.upper()} - SNR {snr} - Fold {fold_ix} - AUC {auc_rounded} \n  Fold {fold_ix} - Subs in Test {test_subs} '
    
    # Plot the confusion matrix without showing it
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title(plot_title)
    
    # Save the confusion matrix plot to the current directory
    plot_filename = f'confusion_matrix_{sampling_rate}_SNR_{snr}_Fold_{fold_ix}_noise_{noise_ix}_{signal_dataset}_{noise_dataset}.png'
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to avoid showing the plot

    print(f"Confusion matrix saved as {plot_filename}")



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
        covariance_matrix = data_params['covariance_matrix']
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
    
    return all_data


def evaluate_weights_across_snr(all_snr_data, model_info, snr_levels, sampling_rates, k):
    best_auc = -1
    best_weights = None
    
    auc_scores = {
        'sr1': {snr: -1 for snr in snr_levels},
        'sr2': {snr: -1 for snr in snr_levels},
        'add': {snr: -1 for snr in snr_levels},
        'concat': {snr: -1 for snr in snr_levels},
    }
    
    for snr in snr_levels:
        datasets = all_snr_data[snr]

        for sr, X, y, group in datasets:
            
            print(f"{sr.upper()} - Evaluating SNR level: {snr}")
            fold_scores = fit_svm(X, y, group, model_info[sr][snr], k)
            mean_auc = np.mean(list(fold_scores.values()))
            auc_scores[sr][snr] = mean_auc
            print(f"Mean AUC for sampling rate {sr}: {auc_scores[sr][snr]}")

    for sr in sampling_rates:        
        best_snr = max(auc_scores[sr], key=auc_scores[sr].get)
        best_auc = auc_scores[sr][best_snr]

        best_weights_dir = model_info[sr][best_snr]['weights_dir']
        best_weights_filename = model_info[sr][best_snr]['model_filename']
        best_weights_path = f'{best_weights_dir}/{best_weights_filename}'
        #load best weights path 
        with open(best_weights_path, 'rb') as file:
            best_weights = pickle.load(file)
        
        print(f"{sr.upper()} - Best SNR: {best_snr} - Best AUC: {best_auc}")
    return best_weights, best_auc



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



def get_pca_features(data_df, name, n_comp=160):
    le = LabelEncoder()
    group = le.fit_transform(data_df['subject'])
    y = data_df['target']
    y = np.array([str(entry) for entry in y])
    X = data_df[f'{name}_Window']
    X = np.array([np.array(entry) for entry in X])


    pca = PCA(n_components=160)
    X_pca = pca.fit_transform(X)

    return X_pca, y, group





def sum_features(X1, y1, group1, X2, y2, group2):
    pca_summed = X1 + X2
    n_comp = X1.shape[1]

    assert np.array_equal(group1, group2), 'group labels do not match'
    assert np.array_equal(y1, y2), 'noise labels do not match'
    assert len(pca_summed) == len(y1) == len(y2) == \
    len(group1) == len(group2) == len(X1) == len(X2) == 1600, \
    'there should be 1600 datapoints in all labels and datasets'
    assert pca_summed.shape == (1600, n_comp), 'summed pca features should have dimension 1600 x n_comp'

    y_pca = y1
    group_pca = group1


    return pca_summed, y_pca, group_pca