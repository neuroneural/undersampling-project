import numpy as np
import pandas as pd
from polyssifier import poly
import os

data_info = pd.read_csv('/data/users2/jwardell1/undersampling-project/HCP/csv-files/data_poly_noise.csv', header=None)
file_paths = data_info.iloc[:, :3].values.tolist()
labels = data_info.iloc[:, 3].astype(int).tolist()

file_paths_exist = []
labels_exist = []

for row, label in zip(file_paths, labels):
    all_arrays = []
    ix = 0
    for file_path in row:
        if os.path.exists(file_path):
            print(f"ix - {ix}")
            if label == 0:
                if ix != 2:
                    n = 53
                    triu_indices = np.triu_indices(n)
                    loaded_array = np.load(file_path)
                    loaded_array = loaded_array[triu_indices]
                    print(f'len(loaded_array) - {len(loaded_array)}')
                else:
                    loaded_array = np.load(file_path)
                    fnc1 = loaded_array[53:,:]
                    fnc2 = loaded_array[:53,:]
                    n = 53
                    triu_indices = np.triu_indices(n)
                    fnc1 = fnc1[triu_indices]
                    fnc2 = fnc2[triu_indices]
                    loaded_array = np.concatenate((fnc1, fnc2))

                    print(f'***len(loaded_array) - {len(loaded_array)}')
            else:   
                loaded_array = np.load(file_path)
            

            # Zero padding to match length
            if len(loaded_array) < 2862:
                padded_array = np.pad(loaded_array, (0, 2862 - len(loaded_array)), mode='constant')
                print(f'len(padded_array) - {len(padded_array)}')
                all_arrays.append(padded_array)
            else:
                all_arrays.append(loaded_array)
                print(f'len(loaded_array) - {len(loaded_array)}')
            
        else:
            print(f"File not found: {file_path}")
            break

        ix += 1
    if(len(all_arrays) != 0):
        file_paths_exist.append(np.concatenate(all_arrays))
        labels_exist.append(label)
    

labels = np.array(labels_exist)
data = np.array(file_paths_exist)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

report = poly(data, labels, n_folds=8, scale=True, exclude=['Decision Tree', 'Random Forest'])

report.plot_scores(path='og_data')

