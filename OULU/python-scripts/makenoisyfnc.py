import numpy as np

fnc_paths = "/data/users2/jwardell1/undersampling-project/OULU/txt-files/tc_data.txt"

# Read data from data.txt
with open(fnc_paths, 'r') as file:
    lines = file.readlines()

# Iterate over every two lines
for i in range(0, len(lines), 2):
    # Get the file paths from the current two lines
    file_path_sr1 = lines[i].strip()
    file_path_sr2 = lines[i + 1].strip()
    
    print(f'file_path_sr2 - {file_path_sr2}')
    print(f'file_path_sr1 - {file_path_sr1}')


    try:
       sr1 = np.load(file_path_sr1)
       sr2 = np.load(file_path_sr2)
    except Exception as e:
       print(e)
       continue

    if sr1.shape[0] != 53:
       sr1 = sr1.T

    if sr2.shape[0] != 53:
       sr2 = sr2.T

    print(f'sr1.shape - {sr1.shape}')
    print(f'sr2.shape - {sr2.shape}')
    
    if sr2.shape[1] <= 1:
       continue


    proc_dir = file_path_sr1[:file_path_sr1.rfind('/')]
    var_noise = np.load(f'{proc_dir}/var_noise.npy')
    print(f'var_noise.shape - {var_noise.shape}')

    b_noise = var_noise[:,::3]
    r_noise = var_noise


    sr1_noise = sr1 + r_noise
    sr2_noise = sr2 + b_noise

    fnc1 = sr1_noise @ sr1_noise.T
    fnc2 = sr1_noise @ sr1_noise.T
    fnc_concat = np.concatenate((fnc1, fnc2))


    fnc1_triu = fnc1[np.triu_indices(fnc1.shape[0])]
    fnc2_triu = fnc2[np.triu_indices(fnc2.shape[0])]

    filearr = file_path_sr1.split('/')
    file_name = filearr[0]
    prefix = file_name.split('.')[0]

   

    # Save the concatenated array as sr1sr2concat.npy in the same directory
    np.save(f'{proc_dir}/{prefix}_fnc1_triu_noise.npy', fnc1_triu, allow_pickle=True)
    np.save(f'{proc_dir}/{prefix}_fnc2_triu_noise.npy', fnc2_triu, allow_pickle=True)
    np.save(f'{proc_dir}/{prefix}_fnc_concat_noise.npy', fnc_concat, allow_pickle=True)
