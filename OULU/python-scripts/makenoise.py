import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import graph2adj


import matplotlib.pyplot as plt

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


#TODO - fix this and save it for each sub, check to make sure it hasn't diverged/converged same thing as using a fixed seed
u_rate = 1
g = gk.ringmore(53, 10) 
A = graph2adj(g)

nstd = 1.0
burn = 100
threshold = 0.0001
NOISE_SIZE = 2961*2

num_converged = 0
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
            var_noise = genData(W, rate=u_rate, burnin=burn, ssize=NOISE_SIZE, nstd=nstd)
            np.save(f'{sub_out_dir}/var_noise.npy', var_noise, allow_pickle=True)
            print(f'generated noise for {sub_out_dir}')
            num_converged += 1
            converged_subjects.append(i)
            print('plotting components')
            fig, axes = plt.subplots(10, 6)
            
            j = 0
            for row in range(10):
                for col in range(6):
                    if j >= 53:
                        axes[row,col].axis('off')
                    else:
                        axes[row,col].plot(var_noise[j,:])
                        axes[row,col].axis('off')
                    j += 1

            plt.savefig(f'{sub_out_dir}/noise_comps.png')
            print('saved fig')
        
        except Exception as e:
            print(f'Convergence error while generating matrix for dir {sub_out_dir}, num converged: {num_converged}')
            print(e)
            continue