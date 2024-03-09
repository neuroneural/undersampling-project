import numpy as np
import os

# Load subject IDs from subjects.txt
subject_ids_file = '/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt'
with open(subject_ids_file, 'r') as f:
    subject_ids = f.read().splitlines()

# Min-Max normalization function
def min_max_normalize(matrix):
    min_vals = np.min(matrix, axis=1, keepdims=True)
    max_vals = np.max(matrix, axis=1, keepdims=True)
    return (matrix - min_vals) / (max_vals - min_vals)

# Iterate through each subject
for subject_id in subject_ids:
    var_noise_file = f'/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/{subject_id}/processed/var_noise.npy'
    if os.path.exists(var_noise_file):
        var_noise = np.load(var_noise_file)
        # Min-Max normalize each component's time series
        var_noise_scaled = min_max_normalize(var_noise)
        # Save the scaled matrix
        var_noise_scaled_file = f'/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/{subject_id}/processed/var_noise_scaled.npy'
        np.save(var_noise_scaled_file, var_noise_scaled, allow_pickle=True)
    else:
        print(f'File not found for subject ID: {subject_id}')

