import numpy as np

# Read data from data.txt
with open('data.txt', 'r') as file:
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

    # Get the directory path to save the concatenated file
    save_dir = file_path_f2[:file_path_f2.rfind('/')]

    # Save the concatenated array as f1f2concat.npy in the same directory
    np.save(f'{save_dir}/f1f2concat.npy', concat_array)
