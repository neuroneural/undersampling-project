import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
# This iterates over all subjects's
# timeseries, creates segments, and finds the
# upper triangle of the segment's FNC matrix

#Step 1: superimpose each subject's timecourses
# into one record. Section the subject's record into ~100 sections

num_subs = 10
n_trs = 2
n_sections = 100

#DATADIR="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU"
#OUTPUTDIR="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU"
OUTPUTDIR="/data/users2/jwardell1/undersampling-project/OULU"
DATADIR="/data/users2/jwardell1/undersampling-project/OULU"

paths_file = "allsubs_TCs.txt"
subs_file = "subjects.txt"


with open("{}/{}".format(DATADIR, subs_file)) as file:
        sub_ids = file.read().split('\n')


with open("{}/{}".format(DATADIR, paths_file)) as file:
        file_paths = file.read().split(',')

for i in range(num_subs):
    if subject_id == '': 
       continue
    subject_id = sub_ids[i]
    print(f'subject_id-  {subject_id}')
    tr100_path = file_paths[i]
    tr2150_path = file_paths[i+1]

    tr100_tc = scipy.io.loadmat(tr100_path)['TCMax'] #n_regions x n_timepoints
    print(f"tr100_tc.shape {tr100_tc.shape}")

    tr2150_tc = scipy.io.loadmat(tr2150_path)['TCMax'] #n_regions x n_timepoints
    print(f"tr2150_tc.shape {tr2150_tc.shape}")

    #iterate over all points in tr100 data
    n_regions, n_tp_tr100 = tr100_tc.shape
    _, n_tp_tr2150 = tr2150_tc.shape


    full_signal = [[] for j in range(n_regions)]

    for roi in range(n_regions):
        tr2150_ix = 0
        for t in range(n_tp_tr100):
        full_signal[roi].append([tr100_tc[roi][t], 100])
        if t % 21 == 0:
            full_signal[roi].append([tr2150_tc[roi][tr2150_ix], 2150])
            tr2150_ix += 1



    n_elements_persection = len(full_signal[0]) // n_sections
    start_ix = 0
    sections = []

    for section in range(n_sections):
        print(f'section {section}')
        tr100_section = [[] for j in range(n_regions)]
        tr2150_section = [[] for j in range(n_regions)]

        for roi in range(n_regions):
            print(f'roi {roi}')
            n_elements_persection = len(full_signal[roi]) // n_sections
            end_ix = start_ix + n_elements_persection
            print(f'start_ix {start_ix}')
            print(f'end_ix {end_ix}')
            for t in range(start_ix, end_ix):
                print(f't {t}')
                if full_signal[roi][t][1] == 2150:
                    tr2150_section[roi].append(full_signal[roi][t][0])
                else:
                    tr100_section[roi].append(full_signal[roi][t][0])


        tr100_fn = '{}_tr100_section{}.npy'.format(subject_id, section)
        tr100_pth = '{}/{}'.format(OUTPUTDIR, tr100_fn)
        np.save(tr100_pth, np.array(tr100_section))
        sections.append(tr100_pth)

        tr2150_fn = '{}_tr2150_section{}.npy'.format(subject_id, section)
        tr2150_pth = '{}/{}'.format(OUTPUTDIR, tr2150_fn)
        np.save(tr2150_pth, np.array(tr2150_section))
        sections.append(tr2150_pth)

        start_ix = end_ix
        
        

    #Step 2: Find the upper triangle of the fnc matrix for each section and save it

    for section in sections:
        timecourses = np.load(section) # n_regions x time
        zscored_timecourses = zscore(timecourses, axis=1)
        fnc_matrix = np.corrcoef(zscored_timecourses)
        upper = fnc_matrix[np.triu_indices(n_regions)]
        fns = section.split('/')
        fname = fns[len(fns)-1]
        ext = fname.split('.')
        pfx = ext[0]
        print(f'prefix is {pfx}')
        fnc_fname = '{}_triu_fnc.npy'.format(pfx)
        fnc_path = '{}/{}'.format(OUTPUTDIR, fnc_fname)
        np.save(fnc_path, upper)
        plt.imshow(fnc_matrix)
        savename = '{}/{}_fnc.png'.format(OUTPUTDIR, pfx)
        plt.savefig(savename)

