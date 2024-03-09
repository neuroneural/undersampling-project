import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import scipy.io
# This iterates over all subjects's
# timeseries, creates segments, and finds the
# upper triangle of the segment's FNC matrix

#Step 1: superimpose each subject's timecourses
# into one record. Section the subject's record into ~100 sections

num_subs = 10
n_sections = 18

OUTPUTDIR="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU"
DATADIR="/data/users2/jwardell1/undersampling-project/OULU/txt-files"

paths_file = "allsubs_TCs.txt"
subs_file = "subjects.txt"


with open("{}/{}".format(DATADIR, subs_file)) as file:
        sub_ids = file.read().split('\n')


with open("{}/{}".format(DATADIR, paths_file)) as file:
        file_paths = file.read().split(',')

for i in range(num_subs):
    subject_id = sub_ids[i]
    if subject_id == '': 
       continue
    
    print(f'subject_id-  {subject_id}')
    tr2150_path = file_paths[i]
    tr100_path = file_paths[i+1]

    tr100_tc = scipy.io.loadmat(tr100_path)['TCMax'] #n_regions x n_timepoints


    tr2150_tc = scipy.io.loadmat(tr2150_path)['TCMax'] #n_regions x n_timepoints


    if tr100_tc.shape[0] > tr100_tc.shape[1]:
        tr100_tc = tr100_tc.T

    if tr2150_tc.shape[0] > tr2150_tc.shape[1]:
        tr2150_tc = tr2150_tc.T


    print(f"tr100_tc.shape {tr100_tc.shape}")
    print(f"tr2150_tc.shape {tr2150_tc.shape}")


    n_regions, n_tp_tr100 = tr100_tc.shape
    _, n_tp_tr2150 = tr2150_tc.shape

    print(f'n_regions - {n_regions}')
    print(f'n_tp_tr100 - {n_tp_tr100}')
    print(f'n_tp_tr2150 - {n_tp_tr2150}')




    len_tr100_section = n_tp_tr100 // n_sections
    tr100_start_ix = 0
    tr100_end_ix = len_tr100_section
    print(f'len_tr100_section - {len_tr100_section}')
    print(f'tr100_start_ix - {tr100_start_ix}')
    print(f'tr100_end_ix - {tr100_end_ix}')

    len_tr2150_section = n_tp_tr2150 // n_sections
    tr2150_start_ix = 0
    tr2150_end_ix = len_tr2150_section
    print(f'len_tr2150_section - {len_tr2150_section}')
    print(f'tr2150_start_ix - {tr2150_start_ix}')
    print(f'tr2150_end_ix - {tr2150_end_ix}')


    tc_paths = []
    for j in range(n_sections):
        tr100_section = tr100_tc[:,tr100_start_ix:tr100_end_ix]
        tr2150_section = tr2150_tc[:,tr2150_start_ix:tr2150_end_ix]

        print(f'tr100_section.shape - {tr100_section.shape}')
        print(f'tr2150_section.shape - {tr2150_section.shape}')

        tr100_fp = '{}/{}/processed/{}_tr100_section{}.npy'.format(OUTPUTDIR, subject_id, subject_id, j)
        tr2150_fp = '{}/{}/processed/{}_tr2150_section{}.npy'.format(OUTPUTDIR, subject_id, subject_id, j)

        np.save(tr100_fp, tr100_section, allow_pickle=True)
        np.save(tr2150_fp, tr2150_section, allow_pickle=True)

        tc_paths.append(tr100_fp)
        tc_paths.append(tr2150_fp)

        tr100_start_ix = tr100_end_ix
        tr100_end_ix = tr100_end_ix + len_tr100_section
        print(f'tr100_start_ix - {tr100_start_ix}')
        print(f'tr100_end_ix - {tr100_end_ix}')
            
        tr2150_start_ix = tr2150_end_ix
        tr2150_end_ix = tr2150_end_ix + len_tr2150_section
        print(f'tr2150_start_ix - {tr2150_start_ix}')
        print(f'tr2150_end_ix - {tr2150_end_ix}')

    #Step 2: Find the upper triangle of the fnc matrix for each section and save it

    for file in tc_paths:
        timecourses = np.load(file) # n_regions x time
        zscored_timecourses = zscore(timecourses, axis=1)
        fnc_matrix = np.corrcoef(zscored_timecourses)
        upper = fnc_matrix[np.triu_indices(n_regions)]

        fns = file.split('/')
        fname = fns[len(fns)-1]
        ext = fname.split('.')
        pfx = ext[0]
        print(f'prefix is {pfx}')
        fnc_fname = '{}_triu_fnc.npy'.format(pfx)
        fnc_path = '{}/{}/processed/{}'.format(OUTPUTDIR, subject_id, fnc_fname)
        np.save(fnc_path, upper)
        plt.imshow(fnc_matrix)
        savename = '{}/{}/processed/{}_fnc.png'.format(OUTPUTDIR, subject_id, pfx)
        plt.savefig(savename)

    print(f'subject {i+1} of {num_subs} complete')