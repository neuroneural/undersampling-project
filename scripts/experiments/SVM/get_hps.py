import os
import pickle
import numpy as np
import pandas as pd

# Define the datasets and other parameters
signal_datasets = ['OULU', 'HCP']
noise_datasets = ['FBIRN']#, 'COBRE', 'VAR']
sampling_rates = ['sr1', 'sr2', 'add', 'concat']
optuna = True

# Define the noise levels using np.arange
lower = 1.5
upper = 2.5
step = 0.1
noise_levels = np.round(np.arange(lower, upper + step, step), 1)

# Directory where model weights are stored
base_dir = '/data/users2/jwardell1/undersampling-project/assets/model_weights/'

# Create an empty list to store table rows
rows = []

# Loop over all combinations of signal, noise datasets, sampling rates, and noise levels
total_iterations = len(signal_datasets) * len(noise_datasets) * len(sampling_rates) * len(noise_levels)
current_iteration = 0

for signal_ds in signal_datasets:
    for noise_ds in noise_datasets:
        for sr in sampling_rates:
            for noise_level in noise_levels:
                current_iteration += 1
                print(f"Processing {current_iteration}/{total_iterations}: Signal DS = {signal_ds}, Noise DS = {noise_ds}, SR = {sr}, Noise Level = {noise_level}")
                
                # Construct the file path
                if optuna:
                    file_name = f'{sr}_best_model_SNR_{noise_level}_RBF_{signal_ds}_{noise_ds}_optuna.pkl'
                else:
                    file_name = f'{sr}_best_model_SNR_{noise_level}_RBF_{signal_ds}_{noise_ds}.pkl'
                
                file_path = os.path.join(base_dir, signal_ds, 'rbf', file_name)
                
                # Check if the file exists
                if os.path.exists(file_path):
                    print(f"Loading model weights from {file_path}")
                    
                    if optuna:
                        with open(file_path, 'rb') as file:
                            hps_dict = pickle.load(file)
                        
                        C_value = hps_dict['C']
                        gamma_value = hps_dict['gamma']
                        tol_value = hps_dict['tol']
                    else:
                        # Load the model
                        with open(file_path, 'rb') as file:
                            model = pickle.load(file)
                        
                        # Extract model parameters
                        C_value = model.C
                        gamma_value = model.gamma
                        tol_value = model.tol
                    
                    
                    # Append the row to the list
                    rows.append([signal_ds, noise_ds, sr, noise_level, C_value, gamma_value, tol_value])
                else:
                    print(f"File not found: {file_path}")

# Create a DataFrame to store and display the table
df = pd.DataFrame(rows, columns=['SIGNAL DS', 'NOISE DS', 'SR', 'NOISE LEVEL', 'C VALUE', 'GAMMA VALUE', 'TOL VALUE'])

# Save the DataFrame to a CSV file for further inspection
output_file = 'model_weights_inspection_optuna.csv'
df.to_csv(output_file, index=False)

# Print the final DataFrame
print("\nFinal model weights table:")
print(df)
print(f"\nTable saved to {output_file}")

