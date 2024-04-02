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

def save_auc_vs_noise_plots_to_pwd(df, ex_name):
   # Group the dataframe by classifier
   grouped = df.groupby('classifier')

   # Plot AUC vs noise level for each classifier and save the plots to the current working directory
   for name, group in grouped:
      plt.figure(figsize=(10, 6))  # Adjust size of the plot if needed
      for label, data in group.groupby('noise_svar'):
         plt.plot(data['scalar'], data['auc'], marker='o', linestyle='-')
         plt.xlabel('Noise Level (Scalar)')
         plt.ylabel('AUC')
         plt.title(f'{ex_name}- AUC vs Noise Level for {name}')
         plt.legend()
         plt.grid(True)
         plt.savefig(f'{ex_name}_auc_vs_noise_{name}.png')  # Save the plot as an image file in the current working directory
         plt.close()

   # Usage example:
   # save_auc_vs_noise_plots_to_pwd(concat_with_auc)  # Pass the dataframe with AUC values appended

pkl_dir = '/data/users2/jwardell1/undersampling-project/OULU/pkl-files/'

sr1 = pd.read_pickle('sr1.pkl')
sr2 = pd.read_pickle('sr2.pkl')
concat = pd.read_pickle('concat.pkl')

sr1_with_auc = calculate_auc_for_classifiers(sr1)
sr2_with_auc = calculate_auc_for_classifiers(sr2)
concat_with_auc = calculate_auc_for_classifiers(concat)

save_auc_vs_noise_plots_to_pwd(sr1_with_auc, 'SR1')
save_auc_vs_noise_plots_to_pwd(sr2_with_auc, 'S2')
save_auc_vs_noise_plots_to_pwd(concat_with_auc, 'Concat')