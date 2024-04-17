import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def calculate_auc(classifier_df):
# Remove rows with NaN values in 'test_proba'
   classifier_df = classifier_df.dropna(subset=['test_proba'])

   # Loop through each row
   for index, row in classifier_df.iterrows():
      # Extract true labels and predicted probabilities
      true_labels = row['target']
      predicted_proba = row['test_proba']
      if not pd.notna(row['test_proba']).all():
         print(f"Skipping '{row['classifier']}' classifier due to NaN values in 'test_proba'.")
         continue
         
      # Calculate fpr, tpr, and thresholds
      fpr, tpr, thresholds = roc_curve(true_labels, predicted_proba)
      roc_auc = auc(fpr, tpr)

      # Append AUC value to the row
      classifier_df.at[index, 'auc'] = roc_auc

   return classifier_df
   
# Define the plot_save_roc_curves_for_classifiers function
def calculate_auc_for_classifiers(data):
   """
   Calculate AUC values for classifiers in each DataFrame.

   Args:
   - data (DataFrame): DataFrame containing classifier data.

   Returns:
   - data_with_auc (DataFrame): DataFrame with appended 'auc' column containing AUC values.
   """
   data_with_auc = pd.DataFrame()

   for classifier_name, classifier_df in data.groupby('classifier'):
   # Skip classifiers with NaN values in 'test_proba'
      if classifier_df['test_proba'].isnull().any():
         print(f"Skipping '{classifier_name}' classifier due to NaN values in 'test_proba'.")
         continue
   
      # Calculate AUC for the classifier
      classifier_df = calculate_auc(classifier_df)
      data_with_auc = pd.concat([data_with_auc, classifier_df])

   return data_with_auc
   
# Usage example:
# Load your dataframes sr1, sr2, and concat
# sr1_with_auc = calculate_auc_for_classifiers(sr1)
# sr2_with_auc = calculate_auc_for_classifiers(sr2)
# concat_with_auc = calculate_auc_for_classifiers(concat)

import os

def save_auc_vs_noise_plots_to_pwd(df1, df2, df3, df4):
   # Group the dataframes by classifier
   grouped1 = df1.groupby('classifier')
   grouped2 = df2.groupby('classifier')
   grouped3 = df3.groupby('classifier')
   grouped4 = df4.groupby('classifier')


   # Plot AUC vs noise level for each classifier and sampling rate and save the plots to the current working directory
   for name, group1 in grouped1:
      plt.figure(figsize=(10, 6))  # Adjust size of the plot if needed
      for idx, (label, data1) in enumerate(group1.groupby('nstd')):
         plt.plot(data1['snr'], data1['auc'], marker='o', linestyle='-', color='green', label='TR=100ms')
      
      # Add data from df2
      for idx, (label, data2) in enumerate(grouped2.get_group(name).groupby('nstd')):
         plt.plot(data2['snr'], data2['auc'], marker='s', linestyle='--', color='red', label='TR=2150ms')

      # Add data from df3
      for idx, (label, data3) in enumerate(grouped3.get_group(name).groupby('nstd')):
         plt.plot(data3['snr'], data3['auc'], marker='^', linestyle='-.', color='blue', label='Concat')
      
      # Add data from df4
      for idx, (label, data4) in enumerate(grouped4.get_group(name).groupby('nstd')):
         plt.plot(data4['snr'], data4['auc'], marker='x', linestyle=':', color='purple', label='Add')

      plt.xlabel('SNR')
      plt.ylabel('AUC')
      plt.title(f'AUC vs SNR for {name}')
      plt.legend()
      plt.grid(True)
      plt.savefig(f'add_all_auc_vs_snr_{name}.png')  # Save the plot as an image file in the current working directory
      plt.close()

# Usage example:
# save_auc_vs_noise_plots_to_pwd(sr1_with_auc, sr2_with_auc, concat_with_auc, 'Combined')
import matplotlib.pyplot as plt
import numpy as np

def plot_boxplot_for_classifiers(dataframes):
    classifier_abbreviations = {
        'Multilayer Perceptron': 'MLP',
        'Logistic Regression': 'LR',
        'Linear SVM': 'LSVM',
        'SVM': 'SVM',
        'Naive Bayes': 'NB'
    }

    for df in dataframes:
        grouped_df = df.groupby(['nstd', 'burnin', 'snr', 'scalar'])

        for group_name, group_data in grouped_df:
            if group_data['test_scores'].isnull().any():
                continue

            n_std_value, burnin_value, snr_value, scalar_value = group_name
            
            # Convert 'test_scores' column to a NumPy array
            test_scores_array = group_data['test_scores'].to_numpy()
            
            # Convert test_scores to a list of NumPy arrays
            all_test_scores = [np.array(scores) for scores in test_scores_array]

            plt.figure(figsize=(12, 8))
            plt.boxplot(all_test_scores)
            plt.xlabel('Classifier')
            plt.ylabel('Test Scores')
            plt.title(f'Box and Whisker Plot of Test Scores for All Models\n{n_std_value=}, {burnin_value=}, {snr_value=}, {scalar_value=}')
            plt.xticks(ticks=range(1, len(classifier_abbreviations) + 1), labels=classifier_abbreviations.values())

            filename = f'Boxplot_nstd_{n_std_value}_burnin_{burnin_value}_snr_{snr_value}_scalar_{scalar_value}.png'
            plt.savefig(filename)
            plt.close()



import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_accuracy(target, predictions):
    # Ensure the target and predictions arrays have the same length
    if len(target) != len(predictions):
        raise ValueError("Length of target and predictions arrays must be the same")

    # Count the number of correct predictions
    correct_predictions = sum(target[i] == predictions[i] for i in range(len(target)))

    # If the length of target is zero, return zero accuracy
    if len(target) == 0:
        return 0.0

    # Calculate accuracy
    accuracy = correct_predictions / len(target)

    return accuracy

import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_bars(dataframes, output_dir):
    short = ['MLP','LR','SVM','NB']
    classifiers = ['Multilayer Perceptron', 'Logistic Regression', 'SVM', 'Naive Bayes']
    sampling_rates = ['sr1', 'sr2', 'concat', 'add']
    snr_values = sorted(dataframes[0]['snr'].unique())
    
    for snr in snr_values:
        fig, axs = plt.subplots(1, len(sampling_rates), figsize=(15, 5), sharey=True)
        accuracy_values = {}
        for i, sr in enumerate(sampling_rates):
            
            for clf in classifiers:
                if clf == 'Linear SVM':
                    continue
                accuracies = []
                for df in dataframes:
                    df_snr = df[df['snr'] == snr]
                    if not df_snr.empty:
                        clf_data = df_snr[df_snr['classifier'] == clf]
                        if not clf_data.empty:
                           target = clf_data['target'].iloc[0]
                           predictions = clf_data['predictions'].iloc[0]
                           accuracy = calculate_accuracy(target, predictions)
                           accuracies.append(accuracy)
               
                accuracy_values[sr] = accuracies

            axs[i].bar(short, accuracy_values[sr])
            axs[i].set_title(f'Sampling Rate: {sr}')
            axs[i].set_xlabel('Classifier')
            axs[i].set_ylabel('Accuracy')
            axs[i].set_ylim(0, 1)
        fig.suptitle(f'SNR: {snr}', fontsize=16)
        plt.tight_layout()
        filename = f'accuracy_plot_SNR_{snr}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()





pkl_dir = '/data/users2/jwardell1/undersampling-project/OULU/pkl-files'

sr1 = pd.read_pickle(f'{pkl_dir}/sr1.pkl')
sr2 = pd.read_pickle(f'{pkl_dir}/sr2.pkl')
concat = pd.read_pickle(f'{pkl_dir}/concat.pkl')
add = pd.read_pickle(f'{pkl_dir}/add.pkl')


dataframes = [sr1, sr2, concat, add]
output_dir = "."
#plot_boxplot_for_classifiers(dataframes)
plot_accuracy_bars(dataframes, '.')


'''
sr1_with_auc = calculate_auc_for_classifiers(sr1)
sr2_with_auc = calculate_auc_for_classifiers(sr2)
concat_with_auc = calculate_auc_for_classifiers(concat)
add_with_auc = calculate_auc_for_classifiers(add)

#save_auc_vs_noise_plots_to_pwd(sr1_with_auc, 'SR1')
#save_auc_vs_noise_plots_to_pwd(sr2_with_auc, 'S2')
#save_auc_vs_noise_plots_to_pwd(concat_with_auc, 'Concat')

save_auc_vs_noise_plots_to_pwd(sr1_with_auc, sr2_with_auc, concat_with_auc, add_with_auc)

'''







