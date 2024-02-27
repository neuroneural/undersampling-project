import numpy as np

fnc_paths = "/data/users2/jwardell1/undersampling-project/HCP/shell-scripts/tc_data.txt"

# Read data from data.txt
with open(fnc_paths, 'r') as file:
    lines = file.readlines()

# Iterate over every two lines
for i in range(0, len(lines)):
    # Get the file paths from the current two lines
    file_path = lines[i].strip()


    # Load the data from the two files as numpy arrays
    try:
        tc_data = np.load(file_path)
    except Exception as e:
        print(e)
        continue

    print(f'tc_data.shape - {tc_data.shape}')
    tc_data = tc_data.T
    print(f'tc_data.shape - {tc_data.shape}')

    proc_dir = file_path[:file_path.rfind('/')]
    var_noise = np.load(f'{proc_dir}/var_noise.npy')
    print(f'var_noise.shape - {var_nosie.shape}')

    sr1 = tc_data
    sr2 = tc_data[:,::3]

    b_noise = var_noise[:,::3]
    r_noise = var_noise


   sr1_noise = sr1 + r_noise
   sr2_noise = sr2 + b_noise

   fnc1 = sr1_noise @ sr1_noise.T
   fnc2 = sr1_noise @ sr1_noise.T
   fnc_concat = np.concatenate((fnc1, fnc2))


   fnc1_triu = fnc1[np.triu_indices(fnc1.shape[0])]
   fnc2_triu = fnc2[np.triu_indices(fnc2.shape[0])]
   fnc_concat = fnc_concat.flatten()

    # Save the concatenated array as f1f2concat.npy in the same directory
    np.save(f'{proc_dir}/fnc1_triu.npy', fnc1_triu, allow_pickle=True)
    np.save(f'{proc_dir}/fnc2_triu.npy', fnc2_triu, allow_pickle=True)
    np.save(f'{proc_dir}/fnc_concat.npy', fnc_concat, allow_pickle=True)
