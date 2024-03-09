import numpy as np

# Read data from data.txt
with open('/data/users2/jwardell1/undersampling-project/OULU/txt-files/data_concat.txt', 'r') as file:
    lines = file.readlines()

# Iterate over every two lines
for i in range(0, len(lines), 2):
    # Get the file paths from the current two lines
    file_path_f2 = lines[i].strip()
    file_path_f1 = lines[i + 1].strip()

    # Load the data from the two files as numpy arrays
    f2 = np.load(file_path_f2)
    f1 = np.load(file_path_f1)

    # Concatenate the two arrays
    concat_array = np.concatenate((f2, f1))
    concat_array = concat_array.flatten()

    # Get the directory path to save the concatenated file
    save_dir = file_path_f2[:file_path_f2.rfind('/')]

    #/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/20150210/processed/20150210_tr2150_section6_triu_fnc.npy
    dirs = file_path_f2.split('/')
    prefix = dirs[len(dirs)-1]
    info = prefix.split('_')
    sub_id = info[0]
    section = info[2]
    

    # Save the concatenated array as f1f2concat.npy in the same directory
    #{subject}_concat_section{i}_fnc.npy
    np.save(f'{save_dir}/{sub_id}_concat_{section}_fnc.npy', concat_array)
    print(f'\t\tsubject {sub_id}  saved at {save_dir}/{sub_id}_concat_{section}_fnc.npy')
