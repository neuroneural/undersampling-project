import numpy as np
import pandas as pd
from polyssifier import poly_subject
import os
import logging  # Import the logging module
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import graph2adj
import uuid  # Make sure to import the uuid module

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


def drawsamplesLG(A, nstd=0.1, samples=100):
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




#Step 0: Iterate through values for n_std, burnin, noise_svar
n_std = [1e-4, 1e-3, 1e-2, 1e-1]
burnin = [50, 100, 125, 175]
noise_svar = [1e-4, 1e-3, 1e-2, 1e-1]
NOISE_SIZE=2961*2
subjects = [20150210, 20150417, 20150428, 20151110, 20151127, 
            20150410, 20150421, 20151030, 20151117, 20151204]



for std in n_std:
    for burn in burnin:
        for noise in noise_svar:
            u_rate = 1
            g = gk.ringmore(53, 10)
            A = graph2adj(g)

            #Step 1: Load data, compute noise, perform zscoring, add noise to loaded data
            num_converged = 0
            noises = dict()
            
            with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/sub_out_dirs.txt', 'r') as file:
                lines = file.readlines()

            for i in range(len(lines)):
                sub_out_dir = lines[i].strip()
                try:
                    W = create_stable_weighted_matrix(A, threshold=0.001, powers=[2])
                    var_noise = genData(W, rate=u_rate, burnin=burn, ssize=NOISE_SIZE, noise=noise, nstd=std)
                except:
                    print(f'convergence error while generating matrix for dir {sub_out_dir}')
                    continue
                
                #TODO - fix this part to keep generating noise til all have converged
                num_converged += 1

                #zscore var_noise
                mean = np.mean(var_noise, axis=1, keepdims=True)
                std = np.std(var_noise, axis=1, keepdims=True)
                var_noise = (var_noise - mean) / std

                noises[subjects[i]] = var_noise


            tc_sr1 = dict()
            tc_sr2 = dict()
            tc_sr1_noise = dict()
            tc_sr2_noise = dict()
            with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/tc_data.txt', 'r') as tc_data:
                lines = tc_data.readlines()
            
            for i in range(len(lines)):
                file_path_sr1 = lines[i].strip()
                file_path_sr2 = lines[i + 1].strip()
                print(f'file_path_sr2 - {file_path_sr2}') #convert to logger
                print(f'file_path_sr1 - {file_path_sr1}') #convert to logger
                try:
                    sr1 = np.load(file_path_sr1)
                    sr2 = np.load(file_path_sr2)
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

                print(f'sr1.shape - {sr1.shape}')#convert to logger
                print(f'sr2.shape - {sr2.shape}')#convert to logger

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
                tc_sr1_noise[subjects[i//2]] = sr1 + var_noise[:,::33]

            #TODO- Step 2: Perform windowing on noise/non-noise data
            

#TODO- Step 3: Run polyssifier on windowed data