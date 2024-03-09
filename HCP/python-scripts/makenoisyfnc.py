import numpy as np
import scipy.io

fnc_paths = "/data/users2/jwardell1/undersampling-project/HCP/txt-files/tc_data.txt"

# Read data from data.txt
with open(fnc_paths, 'r') as file:
    lines = file.readlines()

# Iterate over every two lines
for i in range(0, len(lines)):
    # Get the file paths from the current two lines
    file_path = lines[i].strip()


    # Load the data from the two files as numpy arrays
    try:
        tc_data = scipy.io.loadmat(file_path)['TCMax']
    except Exception as e:
        print(e)
        continue

    print(f'tc_data.shape - {tc_data.shape}')
    tc_data = tc_data.T
    mean = np.mean(tc_data, axis=1, keepdims=True)
    std = np.std(tc_data, axis=1, keepdims=True)
    tc_data = (tc_data - mean) / std
    print(f'tc_data.shape - {tc_data.shape}')

    proc_dir = file_path[:file_path.rfind('/')]
    var_noise = np.load(f'{proc_dir}/var_noise.npy')
    mean = np.mean(var_noise, axis=1, keepdims=True)
    std = np.std(var_noise, axis=1, keepdims=True)
    var_noise = (var_noise - mean) / std
    print(f'var_noise.shape - {var_noise.shape}')

    sr1 = tc_data
    sr2 = tc_data[:,::3]

    b_noise = var_noise[:,::3]
    r_noise = var_noise

    
    

    # Check and pad arrays if dimensions don't match
    if sr1.shape[1] != r_noise.shape[1]:
        if sr1.shape[1] > r_noise.shape[1]:
            padding = np.zeros((sr1.shape[0], sr1.shape[1] - r_noise.shape[1]))
            r_noise = np.hstack((r_noise, padding))
        else:
            padding = np.zeros((r_noise.shape[0], r_noise.shape[1] - sr1.shape[1]))
            sr1 = np.hstack((sr1, padding))

    if sr2.shape[1] != b_noise.shape[1]:
        if sr2.shape[1] > b_noise.shape[1]:
            padding = np.zeros((sr2.shape[0], sr2.shape[1] - b_noise.shape[1]))
            b_noise = np.hstack((b_noise, padding))
        else:
            padding = np.zeros((b_noise.shape[0], b_noise.shape[1] - sr2.shape[1]))
            sr2 = np.hstack((sr2, padding))

    sr1_noise = sr1 + r_noise/100
    sr2_noise = sr2 + b_noise/100

    fnc1 =  np.corrcoef(sr1_noise)
    fnc2 = np.corrcoef(sr2_noise)
    fnc_concat = np.concatenate((fnc1, fnc2))


    fnc1_triu = fnc1[np.triu_indices(fnc1.shape[0])]
    fnc2_triu = fnc2[np.triu_indices(fnc2.shape[0])]
    fnc_concat = np.concatenate((fnc1_triu, fnc2_triu))

    # Save the concatenated array as f1f2concat.npy in the same directory
    np.save(f'{proc_dir}/fnc1_triu.npy', fnc1_triu, allow_pickle=True)
    np.save(f'{proc_dir}/fnc2_triu.npy', fnc2_triu, allow_pickle=True)
    np.save(f'{proc_dir}/fnc_concat.npy', fnc_concat, allow_pickle=True)
