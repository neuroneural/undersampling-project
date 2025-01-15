import itertools
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io
import mne
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle


srs = ['sr1', 'sr2', 'add', 'concat']
dataset = 'oulu'
conn_thresh = 3.25

comp_labels = ['Caudate1', 'Hypothalamus', 'Putamen', 'Caudate2', 'Thalamus', 'STG', 'MTG1', 'PoCG1', 'L PoCG', 'ParaCL1', 'R PoCG', 'SPL1', 'ParaCL2', 'PreCG', 'SPL2', 'PoCG2', 'CalcarineG', 'MOG', 'MTG2', 'Cuneus', 'R MOG', 'FusiformG', 'IOG', 'LingualG', 'MTG3', 'IPL1', 'Insula', 'SMFG', 'IFG1', 'R IFG', 'MiFG1', 'IPL2', 'R IPL', 'SMA', 'SFG', 'MiFG', 'HiPP1', 'L IPL', 'MCC', 'IFG2', 'MiFG2', 'HiPP2', 'Precuneus1', 'Precuneus2', 'ACC1', 'PCC1', 'ACC2', 'Precuneus3', 'PCC2', 'CB1', 'CB2', 'CB3', 'CB4']


edges = 1378

for sr in srs:
    coeffs = np.load(f'coeff_{sr}.npy', allow_pickle=True)


    if sr == 'concat': 
        coeffs1 = coeffs[:1431]
        coeffs2 = coeffs[1431:]
        conn1 = np.zeros((53,53))
        triu_indices = np.triu_indices(53)
        conn1[triu_indices] = coeffs1
        conn1 += conn1.T
        np.fill_diagonal(conn1, 0)
        conn1 = sc.stats.zscore(conn1)

        conn2 = np.zeros((53,53))
        conn2[triu_indices] = coeffs2
        conn2 += conn2.T
        np.fill_diagonal(conn2, 0)
        conn2 = sc.stats.zscore(conn2)

    else:
        conn = np.zeros((53, 53))
        triu_indices = np.triu_indices(53)
        conn[triu_indices] = coeffs
        conn += conn.T
        np.fill_diagonal(conn, 0)
        conn = sc.stats.zscore(conn)




    group_bounds = [0, 5, 7, 16, 25, 42, 49]
    len_SC = 5
    len_AUD = 2
    len_SM = 9
    len_VIS = 9
    len_CC = 17
    len_DM = 7
    len_CB = 4

    node_angles = circular_layout(comp_labels, comp_labels, start_pos=90,
                        group_boundaries=group_bounds)

    cmap = cm.get_cmap('Spectral', 7)

    node_colors = list(itertools.repeat(cmap(0), len_SC)) + list(itertools.repeat(cmap(1),len_AUD)) + list(itertools.repeat(cmap(2),len_SM)) + list(itertools.repeat(cmap(3), len_VIS)) + list(itertools.repeat(cmap(4),len_CC)) + list(itertools.repeat(cmap(5), len_DM)) + list(itertools.repeat(cmap(6), len_CB))


    nedges = np.where(np.abs(conn) > conn_thresh)[0].shape[0]

    if nedges == 0:
        plot_connectivity_circle(np.zeros((53,53)), comp_labels,
                    node_angles=node_angles, node_colors=node_colors, colormap = 'bwr', 
                    vmin=-1, vmax=1, facecolor = 'white', textcolor = 'black', linewidth = 3, 
                    colorbar=False, fontsize_names=12, show = True)
    else:
        if sr != 'concat':
            plot_connectivity_circle(conn, comp_labels, n_lines = nedges,
                    node_angles=node_angles, node_colors=node_colors, colormap = 'bwr', 
                    facecolor = 'white', textcolor = 'black', linewidth = 3, colorbar=False, 
                    fontsize_names=12, show = True)
            plt.title(f'Significant Weights Learned by LR for {sr.upper()} @ SNR = 1.5, {dataset.upper()} Dataset')
            plt.tight_layout()
            plt.savefig(f'undersampling-project/circos_oulu_{sr}_1.5.png')

            plt.clf()
        else: 
            nedges = np.where(np.abs(conn1) > conn_thresh)[0].shape[0]
            plot_connectivity_circle(conn1, comp_labels, n_lines = nedges,
                    node_angles=node_angles, node_colors=node_colors, colormap = 'bwr', 
                    facecolor = 'white', textcolor = 'black', linewidth = 3, colorbar=False, 
                    fontsize_names=12, show = True)
            
            plt.title(f'Significant Weights Learned by LR for {sr.upper()} (SR1 part) @ SNR = 1.5, {dataset.upper()} Dataset')
            plt.tight_layout()
            plt.savefig(f'undersampling-project/circos_oulu_{sr}_1_1.5.png')

            plt.clf()
            nedges = np.where(np.abs(conn2) > conn_thresh)[0].shape[0]
            plot_connectivity_circle(conn2, comp_labels, n_lines = nedges,
                    node_angles=node_angles, node_colors=node_colors, colormap = 'bwr', 
                    facecolor = 'white', textcolor = 'black', linewidth = 3, colorbar=False, 
                    fontsize_names=12, show = True)
            
            plt.title(f'Significant Weights Learned by LR for {sr.upper()} (SR2 part) @ SNR = 1.5, {dataset.upper()} Dataset')
            plt.tight_layout()
            plt.savefig(f'undersampling-project/circos_oulu_{sr}_2_1.5.png')

            plt.clf()
