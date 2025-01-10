import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from multiprocessing import Pool, cpu_count
from matplotlib.gridspec import GridSpec

def load_and_assign_sampling_rate(rate, files):
    """ Load files for a given rate and assign sampling rate. """
    file_list = glob.glob(files)
    if file_list:
        print(f"Found {len(file_list)} files for sampling rate {rate.upper()}.")
        return pd.concat((pd.read_pickle(f).assign(sampling_rate=rate) for f in file_list), ignore_index=True)
    else:
        print(f"No files found for sampling rate {rate.upper()}.")
    return pd.DataFrame()

def load_data(signal_dataset, noise_dataset, date):
    """ Function to load and concatenate pkl files for a dataset. """
    print(f"Loading data for {signal_dataset.upper()} dataset and {noise_dataset.upper()} noise (Date: {date})")
    pkl_dir = f'/data/users2/jwardell1/undersampling-project/{signal_dataset.upper()}/pkl-files/{noise_dataset.upper()}/{date}'

    joined_files = {rate: os.path.join(pkl_dir, f'{rate}_*.pkl') for rate in ['sr1', 'sr2', 'concat', 'add']}
    dfs = []

    with Pool(cpu_count()) as pool:
        results = [pool.apply_async(load_and_assign_sampling_rate, args=(rate, files)) for rate, files in joined_files.items()]
        for result in results:
            dfs.append(result.get())

    result_df = pd.concat(dfs, ignore_index=True)
    result_df = result_df.rename(columns={'roc': 'auc'})
    print(f"Loaded {len(result_df)} rows of data.")
    return result_df

def preprocess_data(result_df, signal_dataset):
    """ Standardize column names and dataset-specific settings. """
    print(f"Preprocessing data for {signal_dataset.upper()} dataset.")

    result_df = result_df.dropna(subset=['test_scores'])
    print(f"Splitting test_scores into individual rows using vectorized operations.")

    # Use a vectorized approach to split test_scores
    exploded_df = result_df.explode('test_scores')
    exploded_df['cv_ix'] = exploded_df.groupby(level=0).cumcount()
    exploded_df = exploded_df.rename(columns={'test_scores': 'auc'})

    result_df = exploded_df.reset_index(drop=True)

    result_df['classifier'] = result_df['classifier'].replace({
        'Logistic Regression': 'LR',
        'Multilayer Perceptron': 'MLP',
        'SVM': 'SVM',
        'Naive Bayes': 'NB'
    })

    result_df['sampling_rate'] = result_df['sampling_rate'].replace({'add': 'Add', 'concat': 'Concat'})

    if signal_dataset == 'oulu':
        result_df['sampling_rate'] = result_df['sampling_rate'].replace({'sr1': 'TR100', 'sr2': 'TR2150'})
    if signal_dataset == 'hcp':
        result_df['sampling_rate'] = result_df['sampling_rate'].replace({'sr1': 'SR1', 'sr2': 'SR2'})

    result_df = result_df[['snr', 'cv_ix', 'auc', 'sampling_rate', 'noise_no', 'classifier']]
    result_df = result_df[result_df['snr'] <= 3]
    print(f"Final preprocessed data contains {len(result_df)} rows.")
    return result_df



def plot_combined_results(fbirn_data, cobre_data):
    """ Generate combined plots for FBIRN (Anomaly Type I) and COBRE (Anomaly Type II). """
    print("Generating combined plots...")
    
    # Define figure and GridSpec with extra space after the second row
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(5, 4, height_ratios=[1, 1, 0.2, 1, 1])  # Small row (0.2) for space

    classifiers = ['LR', 'MLP', 'SVM', 'NB']
    datasets = ['FBIRN', 'COBRE']
    signal_labels = ['Multi-Scale (OULU)', 'Single-Scale (HCP)']

    sampling_rates_oulu = ['TR100', 'TR2150', 'Add', 'Concat']
    sampling_rates_hcp = ['SR1', 'SR2', 'Add', 'Concat']

    palettes_oulu = {rate: plt.cm.tab20(i) for i, rate in enumerate(sampling_rates_oulu)}
    palettes_hcp = {rate: plt.cm.tab20(i) for i, rate in enumerate(sampling_rates_hcp)}

    def plot_subplot(ax, data, title, palette, sampling_rates, row, signal):
        summary_df = data.groupby(['snr', 'sampling_rate']).agg(
            mean_auc=('auc', 'mean'),
            std_auc=('auc', 'std'),
            count=('auc', 'count')
        ).reset_index()
        summary_df['se_auc'] = summary_df['std_auc'] / np.sqrt(summary_df['count'])

        for rate in sampling_rates:
            rate_df = summary_df[summary_df['sampling_rate'] == rate]
            rate_df['mean_auc'] = pd.to_numeric(rate_df['mean_auc'], errors='coerce')
            rate_df['se_auc'] = pd.to_numeric(rate_df['se_auc'], errors='coerce')
            if not rate_df.empty:
                ax.plot(rate_df['snr'], rate_df['mean_auc'], marker='o', label=rate, color=palette[rate])
                ax.fill_between(rate_df['snr'], rate_df['mean_auc'] - rate_df['se_auc'],
                                rate_df['mean_auc'] + rate_df['se_auc'], alpha=0.2, color=palette[rate])

        add_label = (row == 0 and signal == 'oulu') or (row == 1 and signal == 'oulu')
        if add_label:
            ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle='--')
        ax.set_ylim(0.5, 1.0)  # Set y-axis limits here

    # Generate plots
    for row, dataset in enumerate(datasets):
        data = fbirn_data if dataset == 'FBIRN' else cobre_data
        for col, classifier in enumerate(classifiers):
            for i, signal in enumerate(['oulu', 'hcp']):
                grid_row = row * 2 + i if row < 1 else row * 2 + i + 1  # Skip the empty row
                ax = fig.add_subplot(gs[grid_row, col])
                subset = data[signal]
                filtered_data = subset[subset['classifier'] == classifier]
                title = classifier
                if col == 0:  # Add multi-scale/single-scale labels
                    ax.set_ylabel(signal_labels[i], fontsize=14, labelpad=10)
                palette = palettes_oulu if signal == 'oulu' else palettes_hcp
                sampling_rates = sampling_rates_oulu if signal == 'oulu' else sampling_rates_hcp
                plot_subplot(ax, filtered_data, title, palette, sampling_rates, row, signal)

    # Remove redundant labels
    for ax in fig.get_axes():
        pos = ax.get_subplotspec().get_gridspec()
        if ax.get_subplotspec().rowspan.stop < pos.nrows:  # Not last row
            ax.set_xticklabels([])
        if ax.get_subplotspec().colspan.start > 0:  # Not first column
            ax.set_yticklabels([])

    # Add clean titles for Anomaly Types
    fig.text(0.98, 0.98, 'Anomaly Type I', ha='right', fontsize=16, fontweight='bold')  # Adjusted higher
    fig.text(0.98, 0.49, 'Anomaly Type II', ha='right', fontsize=16, fontweight='bold')

    # Add global caption at the bottom
    fig.text(0.5, 0.01, 'Figure 1: Comparison of AUC scores across classifiers and sampling rates for Anomaly Types I and II.',
             ha='center', fontsize=12, fontstyle='italic')

    # Manually adjust margins to make space for the caption
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

    # Global labels
    fig.text(0.015, 0.5, 'AUC', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.04, 'SNR', ha='center', fontsize=16)

    plt.savefig('combined_anomaly_plot_highres.png', dpi=100)
    print("Plot saved successfully.")
    plt.show()


if __name__ == "__main__":
    print("Starting parallel data loading for FBIRN and COBRE...")
    
    date = '12-16'
    # Load FBIRN data
    fbirn_data = {
        'oulu': load_data('oulu', 'fbirn', date='11-25'),
        'hcp': load_data('hcp', 'fbirn', date='11-19')
    }

    # Load COBRE data
    cobre_data = {
        'oulu': load_data('oulu', 'cobre', date='11-18'),
        'hcp': load_data('hcp', 'cobre', date='11-20')
    }

    # Preprocess data
    for key in fbirn_data:
        fbirn_data[key] = preprocess_data(fbirn_data[key], key)
        cobre_data[key] = preprocess_data(cobre_data[key], key)

    # Generate combined plot
    plot_combined_results(fbirn_data, cobre_data)

