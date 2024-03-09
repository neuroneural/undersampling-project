import numpy as np

fnc_paths = "/data/users2/jwardell1/undersampling-project/OULU/txt-files/tc_data.txt"
N_SUBS = 10
N_SRS = 2

# Read data from data.txt
with open(fnc_paths, 'r') as file:
    lines = file.readlines()

n_sections = len(lines) / N_SRS / N_SUBS


ix = 0

# Iterate over every two lines
for i in range(0, len(lines), N_SRS):
    print(f'i - {i}')

    if ((i != 0) and (i % 36 == 0)):
       ix = 0


    #if i < 18: 
    #   ix += 1
    #   continue

    print(f'ix = {ix}')

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
       #continue

    if sr1.shape[0] != 53:
       sr1 = sr1.T

    if sr2.shape[0] != 53:
       sr2 = sr2.T


    if sr1.shape[1] > sr2.shape[1]:
       temp = sr2
       sr2 = sr1
       sr1 = temp


    print(f'sr1.shape - {sr1.shape}')
    print(f'sr2.shape - {sr2.shape}')


    proc_dir = file_path_sr1[:file_path_sr1.rfind('/')]
    var_noise = np.load(f'{proc_dir}/var_noise.npy')
    print(f'var_noise.shape - {var_noise.shape}')

    mean = np.mean(var_noise, axis=1, keepdims=True)
    std = np.std(var_noise, axis=1, keepdims=True)
    var_noise = (var_noise - mean) / std

    n_elements_per_section = var_noise.shape[1] // n_sections

   
    start_ix = int(ix * n_elements_per_section)
    end_ix = int(start_ix + n_elements_per_section)

    sampling_factor = int(np.ceil(1 / ( (sr1.shape[1] * n_sections) / var_noise.shape[1] )))

    b_noise = var_noise[:,start_ix:end_ix:sampling_factor]
    r_noise = var_noise[:,start_ix:end_ix]


    sr1_noise = sr1 + b_noise
    sr2_noise = sr2 + r_noise

    fnc1 = np.corrcoef(sr1_noise)
    fnc2 = np.corrcoef(sr2_noise)
    fnc_concat = np.concatenate((fnc1, fnc2))


    fnc1_triu = fnc1[np.triu_indices(fnc1.shape[0])]
    fnc2_triu = fnc2[np.triu_indices(fnc2.shape[0])]

    filearr = file_path_sr1.split('/')
    file_name = filearr[len(filearr)-1]
    prefix = file_name.split('.')[0]
    section_info = prefix.split('_')
    subject_id = section_info[0]
    section = section_info[len(section_info)-1]

   

    # Save the concatenated array as sr1sr2concat.npy in the same directory
    np.save(f'{proc_dir}/{subject_id}_tr2150_{section}_fnc1_triu_noise.npy', fnc1_triu, allow_pickle=True)
    np.save(f'{proc_dir}/{subject_id}_tr100_{section}_fnc2_triu_noise.npy', fnc2_triu, allow_pickle=True)
    np.save(f'{proc_dir}/{subject_id}_concat_{section}_fnc_noise.npy', fnc_concat, allow_pickle=True)

    ix += 1
