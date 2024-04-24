import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator



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




   

def save_auc_vs_noise_plots_to_pwd(df1, df2, df3, df4, lower_limit=None):
   # Group the dataframes by classifier
   grouped1 = df1.groupby('classifier')
   grouped2 = df2.groupby('classifier')
   grouped3 = df3.groupby('classifier')
   grouped4 = df4.groupby('classifier')
   


   # Plot AUC vs noise level for each classifier and sampling rate and save the plots to the current working directory
   for name, group1 in grouped1:
      plt.figure(figsize=(10, 6))  # Adjust size of the plot if needed
      for idx, (label, data1) in enumerate(group1.groupby('nstd')):
         if lower_limit is None:
            plt.plot(data1['snr'], data1['auc'], marker='o', linestyle='-', color='green', label='TR=SR1')
         else:
            data_filtered = data1[data1['snr'] >= lower_limit]
            plt.plot(data_filtered['snr'], data_filtered['auc'], marker='o', linestyle='-', color='green', label='TR=SR1')

         
      
      # Add data from df2
      for idx, (label, data2) in enumerate(grouped2.get_group(name).groupby('nstd')):
         if lower_limit is None:
            plt.plot(data2['snr'], data2['auc'], marker='s', linestyle='--', color='red', label='TR=SR2')
         else:
            data_filtered = data2[data2['snr'] >= lower_limit]
            plt.plot(data_filtered['snr'], data_filtered['auc'], marker='s', linestyle='--', color='red', label='TR=SR2')

      # Add data from df3
      for idx, (label, data3) in enumerate(grouped3.get_group(name).groupby('nstd')):
         if lower_limit is None:
            plt.plot(data3['snr'], data3['auc'], marker='^', linestyle='-.', color='blue', label='Concat')
         else:
            data_filtered = data3[data3['snr'] >= lower_limit]
            plt.plot(data_filtered['snr'], data_filtered['auc'], marker='^', linestyle='-.', color='blue', label='Concat')
      
      # Add data from df4
      for idx, (label, data4) in enumerate(grouped4.get_group(name).groupby('nstd')):
         if lower_limit is None:
            plt.plot(data4['snr'], data4['auc'], marker='x', linestyle=':', color='purple', label='Add')
         else:
            data_filtered = data4[data4['snr'] >= lower_limit]
            plt.plot(data_filtered['snr'], data_filtered['auc'], marker='x', linestyle=':', color='purple', label='Add')


      plt.xlabel('SNR')
      plt.ylabel('AUC')
      plt.title(f'AUC vs SNR for {name}')
      plt.legend()
      plt.grid(True)
      plt.savefig(f'add_all_auc_vs_snr_{name}.png')  # Save the plot as an image file in the current working directory
      plt.close()

# Usage example:
# save_auc_vs_noise_plots_to_pwd(sr1_with_auc, sr2_with_auc, concat_with_auc, 'Combined')


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


def count_winners(dataframes, lower_limit=0):
    classifiers = ['Multilayer Perceptron', 'Logistic Regression', 'SVM', 'Naive Bayes']
    sampling_rates = ['TR=SR1', 'TR=SR2', 'Concat', 'Add']
    snr_values = sorted([snr for snr in dataframes[0]['snr'].unique() if snr >= lower_limit])
    num_wins_per_sr = {
       'TR=SR1' : 0, 
       'TR=SR2' : 0,
       'Concat' : 0,
       'Add' : 0, 
       'Lower Limit' : lower_limit
    }
    for snr in snr_values:
        ix = 0
        sr_with_best_mean_acc = ''
        best_mean_acc = float('-inf')
        
        for df in dataframes:
            sr = sampling_rates[ix]
            accuracies = []

            for clf in classifiers:
               if clf == 'Linear SVM':
                   continue

               df_snr = df[df['snr'] == snr]
               if not df_snr.empty:
                 clf_data = df_snr[df_snr['classifier'] == clf]
                 if not clf_data.empty:
                    target = clf_data['target'].iloc[0]
                    predictions = clf_data['predictions'].iloc[0]
                    accuracy = calculate_accuracy(target, predictions)
                    accuracies.append(accuracy)

            if np.median(accuracies) > best_mean_acc:
               sr_with_best_mean_acc = sr
               best_mean_acc = np.median(accuracies)

            ix += 1

        num_wins_per_sr[sr_with_best_mean_acc] += 1
    return num_wins_per_sr


def plot_accuracy_bars(dataframes, output_dir, lower_limit=0):
    short = ['MLP','LR','SVM','NB']
    classifiers = ['Multilayer Perceptron', 'Logistic Regression', 'SVM', 'Naive Bayes']
    sampling_rates = ['TR=SR1', 'TR=SR2', 'Concat', 'Add']
    snr_values = sorted([snr for snr in dataframes[0]['snr'].unique() if snr >= lower_limit])
    #snr_values = sorted(dataframes[0]['snr'].unique())
    num_wins_per_sr = {
       'TR=SR1' : 0, 
       'TR=SR2' : 0,
       'Concat' : 0,
       'Add' : 0
    }
    for snr in snr_values:
        fig, axs = plt.subplots(1, len(sampling_rates), figsize=(15, 5), sharey=True)
        accuracy_values = {}
        ix = 0
        sr_with_best_mean_acc = ''
        best_mean_acc = float('-inf')
        ax_with_best_mean_acc = 0
        
        for df in dataframes:
            sr = sampling_rates[ix]
            accuracies = []

            for clf in classifiers:
               if clf == 'Linear SVM':
                   continue

               df_snr = df[df['snr'] == snr]
               if not df_snr.empty:
                 clf_data = df_snr[df_snr['classifier'] == clf]
                 if not clf_data.empty:
                    target = clf_data['target'].iloc[0]
                    predictions = clf_data['predictions'].iloc[0]
                    accuracy = calculate_accuracy(target, predictions)
                    accuracies.append(accuracy)

            if np.sum(accuracies) > best_mean_acc:
               sr_with_best_mean_acc = sr
               ax_with_best_mean_acc = ix
               best_mean_acc = np.sum(accuracies)
            accuracy_values[sr] = accuracies

            bars = axs[ix].bar(short, accuracy_values[sr])
            axs[ix].set_title(f'{sr}', pad=20)#'Sampling Rate: {sr}')
            axs[ix].set_xlabel('Classifier')
            axs[ix].set_ylabel('Accuracy')
            axs[ix].set_ylim(0, 1)
            axs[ix].yaxis.set_major_locator(MaxNLocator(nbins=10))
            axs[ix].spines['top'].set_visible(False)
            axs[ix].spines['right'].set_visible(False)
            
            
            acc_ix = 0
            for bar in bars:
               height = bar.get_height()
               axs[ix].annotate(
                    f'{height:.2f}',  # Text format for the value
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Position of the annotation
                    xytext=(0, 3),  # Offset of the annotation from the bar
                    textcoords="offset points",
                    ha='center', va='bottom')  # Alignment of the annotation
               acc_ix += 1
            ix += 1
        
        fig.suptitle(f'SNR: {snr}', fontsize=16, y=0.95)
        axs[ax_with_best_mean_acc].set_title(f'{sr_with_best_mean_acc}*', pad=20)
        num_wins_per_sr[sr_with_best_mean_acc] += 1
        plt.tight_layout()
        filename = f'accuracy_plot_SNR_{snr}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    print(num_wins_per_sr)



def plot_winners(win_df, lower_limit=-2, filename='winners.png'):
   win_df_filtered = win_df[win_df['Lower Limit'] >= lower_limit]
   # Define sampling rates
   sampling_rates = ['TR=SR1', 'TR=SR2', 'Concat', 'Add']

   # Plot line graph for each sampling rate
   plt.figure(figsize=(10, 6))
   colors = ['blue', 'green', 'red', 'orange']
   markers = ['o', 's', 'D', '^']
   linestyles = ['-', '--', '-.', ':']
   for i, rate in enumerate(sampling_rates):
      plt.plot(win_df_filtered['Lower Limit'], win_df_filtered[rate], label=rate, color=colors[i], marker=markers[i], linestyle=linestyles[i])

   # Add labels and legend
   plt.xlabel('Thresholded SNR (Lower Limit)')
   plt.ylabel('Num Wins')
   plt.title('Number of Wins vs Lower Limit')
   plt.legend()
   plt.grid(True)
   plt.savefig(filename)


#pkl_dir = '/data/users2/jwardell1/undersampling-project/HCP/pkl-files'

#sr1 = pd.read_pickle(f'{pkl_dir}/sr1.pkl')
#sr2 = pd.read_pickle(f'{pkl_dir}/sr2.pkl')
#concat = pd.read_pickle(f'{pkl_dir}/concat.pkl')
#add = pd.read_pickle(f'{pkl_dir}/add.pkl')


#dataframes = [sr1, sr2, concat, add]
#output_dir = "."
#plot_boxplot_for_classifiers(dataframes)
#plot_accuracy_bars(dataframes, '.', lower_limit=-2)



#resw = []
#c = np.arange(-2, 2, 0.25)
#for i in c:
#   print(i)
#   winners = count_winners(dataframes, i)
#   print()
#   resw.append(winners)

#win_df = pd.DataFrame(resw)
#print(win_df)
#pd.to_pickle(win_df, 'winners_med.pkl')



#win_df = pd.read_pickle('winners_med.pkl')
#plot_winners(win_df, lower_limit=-2, filename='winners_lim.png')


#sr1_with_auc = calculate_auc_for_classifiers(sr1)
#sr2_with_auc = calculate_auc_for_classifiers(sr2)
#concat_with_auc = calculate_auc_for_classifiers(concat)
#add_with_auc = calculate_auc_for_classifiers(add)

#save_auc_vs_noise_plots_to_pwd(sr1_with_auc, 'SR1')
#save_auc_vs_noise_plots_to_pwd(sr2_with_auc, 'S2')
#save_auc_vs_noise_plots_to_pwd(concat_with_auc, 'Concat')

#save_auc_vs_noise_plots_to_pwd(sr1_with_auc, sr2_with_auc, concat_with_auc, add_with_auc)#, lower_limit=-0.5)



import glob


pkl_dir = '/data/users2/jwardell1/undersampling-project/HCP/pkl-files'
joined_files = os.path.join(pkl_dir, 'sr1_*.pkl')
joined_list = glob.glob(joined_files)
sr1 = pd.concat(map(pd.read_pickle, joined_list), ignore_index=True)


joined_files = os.path.join(pkl_dir, 'sr2_*.pkl')
joined_list = glob.glob(joined_files)
sr2 = pd.concat(map(pd.read_pickle, joined_list), ignore_index=True)

joined_files = os.path.join(pkl_dir, 'concat_*.pkl')
joined_list = glob.glob(joined_files)
concat = pd.concat(map(pd.read_pickle, joined_list), ignore_index=True)


joined_files = os.path.join(pkl_dir, 'add_*.pkl')
joined_list = glob.glob(joined_files)
add = pd.concat(map(pd.read_pickle, joined_list), ignore_index=True)



dataframes = [sr1, sr2, add, concat]
short = ['MLP','LR','SVM','NB']
classifiers = ['Multilayer Perceptron', 'Logistic Regression', 'SVM', 'Naive Bayes']
sampling_rates = ['TR=SR1', 'TR=SR2', 'Concat', 'Add']
snr_values = sorted([snr for snr in dataframes[0]['snr'].unique()])


fig, axs = plt.subplots(1, len(snr_values), figsize=(15, 5), sharey=True, sharex=True)


for snr in snr_values:
    auc_scores = {}
    ix = 0
    for df in dataframes:
        sr = sampling_rates[ix]
        aucs = {'Multilayer Perceptron' : [], 
                'Logistic Regression' : [], 
                'SVM' : [], 
                'Naive Bayes' : []}
        for clf in classifiers:
            df_snr = df[df['snr'] == snr]
            if not df_snr.empty:
                clf_data = df_snr[df_snr['classifier'] == clf]
                aucs[clf].append(clf_data['test_scores'])
        auc_scores[sr] = aucs
        print(short)
        print(auc_scores[sr])


        data_group1 =          

        boxes = axs[ix].boxplot([snr]*4, auc_scores[sr], labels=short)
        axs[ix].set_ylabel('AUC')
        axs[ix].set_ylim(0, 1)

fig.suptitle(f'SNR vs AUC: {snr}', fontsize=16, y=0.95)
plt.tight_layout()
filename = f'AUC_vs_SNR_boxplot.png'
output_dir='.'
plt.savefig(os.path.join(output_dir, filename))
