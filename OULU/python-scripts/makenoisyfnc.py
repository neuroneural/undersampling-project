import numpy as np

fnc_paths = "/data/users2/jwardell1/undersampling-project/OULU/txt-files/tc_data.txt"
N_SUBS = 10
N_SRS = 2

# Read data from data.txt
with open(fnc_paths, 'r') as file:
    lines = file.readlines()

n_sections = len(lines) / N_SRS / N_SUBS


ix = 0
print(f'len lines {len(lines)}')
# Iterate over every two lines
num_files_not_found = 0
for i in range(0, len(lines), N_SRS):
    print(f'i - {i}')

    if ((i != 0) and (i % 160 == 0)):
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
       #print(e)
       #num_files_not_found += 1
       continue

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
    
    if sr2.shape[1] < 1645:
       continue
    
    if sr1.shape[1] < 100:
       continue


    proc_dir = file_path_sr1[:file_path_sr1.rfind('/')]
    var_noise = np.load(f'{proc_dir}/var_noise.npy')
    print(f'var_noise.shape - {var_noise.shape}')

    mean = np.mean(var_noise, axis=1, keepdims=True)
    std = np.std(var_noise, axis=1, keepdims=True)
    var_noise = (var_noise - mean) / std

   #  n_elements_per_section = var_noise.shape[1] // n_sections
    

    window_size = 1645
    stride = 16
    start_ix = int(ix * stride)
    end_ix = int(start_ix + window_size)

    print(f'start_ix - {start_ix}')
    print(f'end_ix - {end_ix}')

    sampling_factor = int(np.ceil(sr2.shape[1] / sr1.shape[1]))

    b_noise = var_noise[:,start_ix:end_ix+55:sampling_factor]
    r_noise = var_noise[:,start_ix:end_ix]
    print(f'b_noise.shape - {b_noise.shape}')
    print(f'r_noise.shape - {r_noise.shape}')

    sr1_noise = sr1 + b_noise/100
    sr2_noise = sr2 + r_noise/100

    fnc1 = np.corrcoef(sr1_noise)
    fnc2 = np.corrcoef(sr2_noise)


    fnc1_triu = fnc1[np.triu_indices(fnc1.shape[0])]
    fnc2_triu = fnc2[np.triu_indices(fnc2.shape[0])]
    fnc_concat = np.concatenate((fnc1_triu, fnc2_triu))

    filearr = file_path_sr1.split('/')
    file_name = filearr[len(filearr)-1]
    prefix = file_name.split('.')[0]
    section_info = prefix.split('_')
    subject_id = section_info[0]
    section = section_info[len(section_info)-1]

   

    # Save the concatenated array as sr1sr2concat.npy in the same directory
    print(f'now saving {proc_dir}/{subject_id}_tr2150_{section}_fnc1_triu_noise.npy')
    np.save(f'{proc_dir}/{subject_id}_tr2150_{section}_fnc1_triu_noise.npy', fnc1_triu, allow_pickle=True)

    print(f'now saving {proc_dir}/{subject_id}_tr100_{section}_fnc2_triu_noise.npy')
    np.save(f'{proc_dir}/{subject_id}_tr100_{section}_fnc2_triu_noise.npy', fnc2_triu, allow_pickle=True)

    print(f'{proc_dir}/{subject_id}_concat_{section}_fnc_noise.npy')
    np.save(f'{proc_dir}/{subject_id}_concat_{section}_fnc_noise.npy', fnc_concat, allow_pickle=True)

    ix += 1

print(2*num_files_not_found)