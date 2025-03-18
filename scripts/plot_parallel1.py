import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from multiprocessing import Pool, cpu_count

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

def plot_combined_results(oulu_df, hcp_df, noise_dataset, kernel_type):
    """ Plot OULU dataset results on top row and HCP dataset results on bottom row. """
    print("Generating plots...")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12), sharey=True, sharex=True)
    classifiers = ['LR', 'MLP', 'SVM', 'NB']
    clf_full = {
        'LR' : 'Logistic Regression',
        'MLP' : 'Multilayer Perceptron',
        'SVM' : 'Support Vector Machine',
        'NB' : 'Naive Bayes',
    }

    # Sampling rates for OULU and HCP datasets
    oulu_sampling_rates = ['TR100', 'TR2150', 'Add', 'Concat']
    hcp_sampling_rates = ['SR1', 'SR2', 'Add', 'Concat']

    # Palettes for OULU and HCP datasets
    oulu_palette = {rate: plt.cm.tab20(i) for i, rate in enumerate(oulu_sampling_rates)}
    hcp_palette = {rate: plt.cm.tab20(i) for i, rate in enumerate(hcp_sampling_rates)}

    def clean_summary_df(df):
        """ Ensure numeric types and drop NaNs. """
        df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
        df['mean_auc'] = pd.to_numeric(df['mean_auc'], errors='coerce')
        df['se_auc'] = pd.to_numeric(df['se_auc'], errors='coerce')
        return df.dropna(subset=['snr', 'mean_auc', 'se_auc'])

    # Plot OULU dataset performance
    print("Plotting OULU dataset performance.")
    for i, classifier in enumerate(classifiers):
        print("np.unique(oulu_df['classifier'])")
        print(np.unique(oulu_df['classifier']))
        data_filtered = oulu_df[oulu_df['classifier'] == classifier]
        print("data_filtered.head()")  # Inspect the DataFrame being plotted
        print(data_filtered.head())  # Inspect the DataFrame being plotted
        print(np.unique(data_filtered['classifier']))  # Inspect the DataFrame being plotted
        summary_df = data_filtered.groupby(['snr', 'sampling_rate']).agg(
            mean_auc=('auc', 'mean'),
            std_auc=('auc', 'std'),
            count=('auc', 'count')
        ).reset_index()

        summary_df['se_auc'] = summary_df['std_auc'] / np.sqrt(summary_df['count'])
        summary_df = clean_summary_df(summary_df)

        print("Unique sampling rates in the data to plot:", summary_df['sampling_rate'].unique())
        print(summary_df.head())  # Inspect the DataFrame being plotted
        print(f'hue_order={oulu_sampling_rates}')
        print(f'palette={oulu_palette}')

        sns.lineplot(
            data=summary_df, x='snr', y='mean_auc', hue='sampling_rate', hue_order=oulu_sampling_rates, 
            palette=oulu_palette, marker='o', ax=axes[0, i]
        )

        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel("")
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("")

        for key, grp in summary_df.groupby('sampling_rate'):
            axes[0, i].fill_between(
                grp['snr'], grp['mean_auc'] - grp['se_auc'], grp['mean_auc'] + grp['se_auc'], 
                alpha=0.2, color=oulu_palette[key]
            )
        axes[0, i].set_title(f"{clf_full[classifiers[i]]}")
        axes[0, i].set_ylim(0.5, 1)  # Setting y-limit

    # Plot HCP dataset performance
    print("Plotting HCP dataset performance.")
    for i, classifier in enumerate(classifiers):
        print("np.unique(hcp_df['classifier'])")
        print(np.unique(hcp_df['classifier']))
        data_filtered = hcp_df[hcp_df['classifier'] == classifier]
        print("data_filtered.head()")  # Inspect the DataFrame being plotted
        print(data_filtered.head())  # Inspect the DataFrame being plotted
        print(np.unique(data_filtered['classifier']))  # Inspect the DataFrame being plotted
        summary_df = data_filtered.groupby(['snr', 'sampling_rate']).agg(
            mean_auc=('auc', 'mean'),
            std_auc=('auc', 'std'),
            count=('auc', 'count')
        ).reset_index()

        summary_df['se_auc'] = summary_df['std_auc'] / np.sqrt(summary_df['count'])
        summary_df = clean_summary_df(summary_df)

        print("Unique sampling rates in the data to plot:", summary_df['sampling_rate'].unique())
        print(summary_df.head())  # Inspect the DataFrame being plotted
        print(f'hue_order={hcp_sampling_rates}')
        print(f'palette={hcp_palette}')

        sns.lineplot(
            data=summary_df, x='snr', y='mean_auc', hue='sampling_rate', hue_order=hcp_sampling_rates, 
            palette=hcp_palette, marker='o', ax=axes[1, i]
        )

        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("")
        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel("")

        for key, grp in summary_df.groupby('sampling_rate'):
            axes[1, i].fill_between(
                grp['snr'], grp['mean_auc'] - grp['se_auc'], grp['mean_auc'] + grp['se_auc'], 
                alpha=0.2, color=hcp_palette[key]
            )
        axes[1, i].set_ylim(0.5, 1)  # Setting y-limit

    # General adjustments
    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, which='both', axis='both', linestyle='--')

    # Label signal datasets further left
    axes[0, 0].annotate("Multi-Scale (OULU)", xy=(-0.25, 0.5), xycoords='axes fraction', fontsize=28, rotation=90,
                        ha='center', va='center', annotation_clip=False)
    axes[1, 0].annotate("Single-Scale (HCP)", xy=(-0.25, 0.5), xycoords='axes fraction', fontsize=28, rotation=90,
                        ha='center', va='center', annotation_clip=False)

    # Set shared axis labels
    fig.text(0.509, 0.15, "SNR", ha='center', fontsize=28)
    fig.text(0.045, 0.57, "AUC", va='center', rotation=90, fontsize=28)

    # Adjust layout to add space at the bottom
    plt.tight_layout(rect=[0.02, 0.15, 0.95, 0.98])  # Bottom margin increased to 0.15


    # Add caption with left justification and center alignment under the figure
    caption = (
        "Figure 1: The mean value of the ROCAUC with shading intervals denoting the standard error are plotted across SNR levels for each classifier. "
        "Results for the Multi-Scale dataset show predictive advantage for combined features, whereas the Single-Scale "
        "dataset shows little to no advantage for combined features. The Add features, the result of summing FNC of the two "
        "rates, shows the most advantage for the Multi-Scale datasets, particularly in Multilayer Perceptron, Logistic Regression, and "
        "Support Vector Machine. The Single-Rate dataset shows the highest advantage in the original sampling rate, followed by "
        "Add and Concat features, with the lowest performance in the undersampled rate."
    )
    plt.figtext(0.5, 0.09, caption, ha='center', va='top', fontsize=16, wrap=True)


    plt.savefig('plot_parallel.png', bbox_inches='tight')
    print("Plot saved successfully.")
    plt.show()

if __name__ == "__main__":
    noise_dataset = 'fbirn'
    kernel_type = 'rbf'
    date = '12-16'
    # Load and preprocess OULU and HCP data
    print("Starting data loading and preprocessing...")
    oulu_df = load_data('oulu', noise_dataset, date='11-25')
    hcp_df = load_data('hcp', noise_dataset, date='11-19')

    oulu_df = preprocess_data(oulu_df, 'oulu')
    hcp_df = preprocess_data(hcp_df, 'hcp')

    # Generate combined plot
    plot_combined_results(oulu_df, hcp_df, noise_dataset, kernel_type)