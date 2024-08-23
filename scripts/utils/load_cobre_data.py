import h5py
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import cholesky, LinAlgError
from textwrap import wrap

# Function to load COBRE data
def load_data(
    dataset_path: str = "/data/users2/ppopov1/datasets/cobre/COBRE_AllData.h5",
    indices_path: str = "/data/users2/ppopov1/datasets/cobre/correct_indices_GSP.csv",
    labels_path: str = "/data/users2/ppopov1/datasets/cobre/labels_COBRE.csv",
):
    """
    Return COBRE data
    """
    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("COBRE_dataset")
    data = np.array(data)
    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    # filter the data: leave only correct components
    data = data[:, idx, :]
    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1
    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]
    return data, labels

# Function to create Toeplitz matrix from covariance matrix
def create_toeplitz(permuted_cov_mat):
    n, _ = permuted_cov_mat.shape
    first_col = permuted_cov_mat[0]
    toeplitz_mat = np.zeros((n, n))
    for i in range(n):
        toeplitz_mat[i] = np.roll(first_col, i)
    return toeplitz_mat

# Load COBRE data
cobre_data, labels = load_data()
subject_ids = range(cobre_data.shape[0])

# Path for saving results
cov_path = '/data/users2/jwardell1/nshor_docker/examples/cobre_project/COV'
if not os.path.exists(cov_path):
    os.makedirs(cov_path)

# Initialize subplot index for covariance and Cholesky matrices
idx = 1
plt.figure(1, figsize=(15, 20))

# Initialize subplot index for time courses
tc_idx = 1
plt.figure(2, figsize=(15, 10))

# Process each subject
for subID in subject_ids:
    tc_data = cobre_data[subID]
    
    # Detrend time course data
    tc_data = signal.detrend(tc_data, axis=1)
    
    # Compute original covariance matrix
    cov_mat_original = np.cov(tc_data, rowvar=False)

    # Permute columns randomly
    permuted_indices = np.random.permutation(tc_data.shape[0])
    permuted_tc_data = tc_data[permuted_indices, :]

    # Compute permuted covariance matrix
    cov_mat_permuted = np.cov(permuted_tc_data, rowvar=False)
    
    # Compute Toeplitz matrix from permuted covariance matrix
    toeplitz_mat = create_toeplitz(cov_mat_permuted)
    
    try:
        # Find Cholesky decomposition of Toeplitz matrix
        chol_matrix = cholesky(toeplitz_mat, lower=True)
    except LinAlgError:
        chol_matrix = None
        print(f'Cholesky decomposition failed for Subject {subID}')
    
    # Save Toeplitz matrix and Cholesky decomposition as .npy
    toeplitz_file = os.path.join(cov_path, f'{subID}_toeplitz.npy')
    np.save(toeplitz_file, toeplitz_mat, allow_pickle=True)
    
    if chol_matrix is not None:
        chol_file = os.path.join(cov_path, f'{subID}_tcov.npy')
        np.save(chol_file, chol_matrix, allow_pickle=True)
    
    # Scale matrices for plotting
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    cov_mat_original_scaled = scaler.fit_transform(cov_mat_original)
    cov_mat_permuted_scaled = scaler.fit_transform(cov_mat_permuted)
    toeplitz_mat_scaled = scaler.fit_transform(toeplitz_mat)
    
    if chol_matrix is not None:
        chol_matrix_scaled = scaler.fit_transform(chol_matrix)
    
    # Plot scaled original and permuted covariance matrices
    plt.figure(1)
    plt.subplot(4, 4, idx)
    plt.imshow(cov_mat_original_scaled, cmap=plt.get_cmap('seismic'), vmin=-1, vmax=1)
    plt.colorbar()
    title = f'Original Covariance Matrix - Subject {subID}'
    plt.title("\n".join(wrap(title, 20)))
    
    plt.subplot(4, 4, idx + 1)
    plt.imshow(cov_mat_permuted_scaled, cmap=plt.get_cmap('seismic'), vmin=-1, vmax=1)
    plt.colorbar()
    title = f'Permuted Covariance Matrix - Subject {subID}'
    plt.title("\n".join(wrap(title, 20)))
    
    # Plot scaled Toeplitz matrix and Cholesky decomposition
    plt.subplot(4, 4, idx + 2)
    plt.imshow(toeplitz_mat_scaled, cmap=plt.get_cmap('seismic'), vmin=-1, vmax=1)
    plt.colorbar()
    title = f'Toeplitz Matrix - Subject {subID}'
    plt.title("\n".join(wrap(title, 20)))
    
    if chol_matrix is not None:
        plt.subplot(4, 4, idx + 3)
        plt.imshow(chol_matrix_scaled, cmap=plt.get_cmap('seismic'), vmin=-1, vmax=1)
        plt.colorbar()
        title = f'Cholesky Decomposition - Subject {subID}'
        plt.title("\n".join(wrap(title, 20)))
    
    # Increment subplot index by 4
    idx += 4

    # Plot original and permuted time course data as heatmap
    plt.figure(2)
    plt.subplot(4, 2, tc_idx)
    plt.imshow(tc_data, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar()
    title = f'Original Time Courses - Subject {subID}'
    plt.title("\n".join(wrap(title, 20)))
    
    plt.subplot(4, 2, tc_idx + 1)
    plt.imshow(permuted_tc_data, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar()
    title = f'Permuted Time Courses - Subject {subID}'
    plt.title("\n".join(wrap(title, 20)))
    
    # Increment time course subplot index by 2
    tc_idx += 2

    print(f'Processed subject {subID}')

# Adjust layout and show the covariance, Toeplitz, and Cholesky matrices plots
plt.figure(1)
plt.tight_layout()
plt.show()

# Adjust layout and show the time course plots
plt.figure(2)
plt.tight_layout()
plt.show()

