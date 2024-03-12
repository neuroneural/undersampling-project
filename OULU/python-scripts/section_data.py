import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import scipy.io
from tqdm import tqdm

# This iterates over all subjects's
# timeseries, creates segments, and finds the
# upper triangle of the segment's FNC matrix

#Step 1: superimpose each subject's timecourses
# into one record. Section the subject's record into ~100 sections

##############LOAD DATA##############

num_subs = 10

OUTPUTDIR="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU"
DATADIR="/data/users2/jwardell1/undersampling-project/OULU/txt-files"

paths_file = "allsubs_TCs.txt"
subs_file = "subjects.txt"

with open("{}/{}".format(DATADIR, subs_file)) as file:
        sub_ids = file.read().split('\n')


with open("{}/{}".format(DATADIR, paths_file)) as file:
        file_paths = file.read().split(',')

##############LOAD DATA##############

for i in range(num_subs):
    subject_id = sub_ids[i]
    if subject_id == '': 
        continue

    ##############Load both TR time courses##############
    print(f'subject_id-  {subject_id}')
    tr2150_path = file_paths[i]
    tr100_path = file_paths[i+1]

    tr100_tc = scipy.io.loadmat(tr100_path)['TCMax'] #n_regions x n_timepoints
    tr2150_tc = scipy.io.loadmat(tr2150_path)['TCMax'] #n_regions x n_timepoints

    ##############Make data n_components x n_timepoints##############
    if tr100_tc.shape[0] > tr100_tc.shape[1]:
        tr100_tc = tr100_tc.T

    if tr2150_tc.shape[0] > tr2150_tc.shape[1]:
        tr2150_tc = tr2150_tc.T

    ##############Zscore time courses for both TRs##############
    mean = np.mean(tr100_tc, axis=1, keepdims=True)
    std = np.std(tr100_tc, axis=1, keepdims=True)
    tr100_tc = (tr100_tc - mean) / std

    mean = np.mean(tr2150_tc, axis=1, keepdims=True)
    std = np.std(tr2150_tc, axis=1, keepdims=True)
    tr2150_tc = (tr2150_tc - mean) / std

    n_regions, n_tp_tr100 = tr100_tc.shape
    _, n_tp_tr2150 = tr2150_tc.shape

    ##############Set values for window size, stride and boundary indices##############
    tr2150_window_size = 100
    tr2150_stride = 1
    n_sections = 80 #n_sections = int((n_tp_tr2150 - tr2150_window_size) / tr2150_stride)
    tr2150_start_ix = 0
    tr2150_end_ix = tr2150_window_size

    tr100_window_size = int((n_tp_tr100 / n_tp_tr2150) * tr2150_window_size)
    tr100_stride = n_tp_tr100 // n_tp_tr2150
    tr100_start_ix = 0
    tr100_end_ix = tr100_window_size

    tc_paths = []
    
    for j in tqdm(range(n_sections), desc=f"Creating Sections"):
        tr100_section = tr100_tc[:, tr100_start_ix:tr100_end_ix]
        tr2150_section = tr2150_tc[:, tr2150_start_ix:tr2150_end_ix]

        tr100_fp = '{}/{}/processed/{}_tr100_section{}.npy'.format(OUTPUTDIR, subject_id, subject_id, j)
        tr2150_fp = '{}/{}/processed/{}_tr2150_section{}.npy'.format(OUTPUTDIR, subject_id, subject_id, j)

        np.save(tr100_fp, tr100_section, allow_pickle=True)
        np.save(tr2150_fp, tr2150_section, allow_pickle=True)

        tc_paths.append(tr100_fp)
        tc_paths.append(tr2150_fp)

        tr100_start_ix += tr100_stride
        tr100_end_ix = tr100_end_ix + tr100_stride
            
        tr2150_start_ix += tr2150_stride
        tr2150_end_ix = tr2150_end_ix + tr2150_stride
        
    #Step 2: Find the upper triangle of the fnc matrix for each section and save it
    
    for file in tqdm(tc_paths, desc=f"Saving FNCs"):
        timecourses = np.load(file) # n_regions x time
        fnc_matrix = np.corrcoef(timecourses)
        upper = fnc_matrix[np.triu_indices(n_regions)]

        fns = file.split('/')
        fname = fns[len(fns)-1]
        ext = fname.split('.')
        pfx = ext[0]
        fnc_fname = '{}_triu_fnc.npy'.format(pfx)
        fnc_path = '{}/{}/processed/{}'.format(OUTPUTDIR, subject_id, fnc_fname)
        np.save(fnc_path, upper)
        plt.imshow(fnc_matrix)
        savename = '{}/{}/processed/{}_fnc.png'.format(OUTPUTDIR, subject_id, pfx)
        plt.savefig(savename)
        

    print(f'subject {i+1} of {num_subs} complete')
