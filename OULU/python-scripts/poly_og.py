import numpy as np
import pandas as pd
from polyssifier import poly
import os

data_info = pd.read_csv('/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly.csv', header=None)
file_paths = data_info.iloc[:, :3].values.tolist()
labels = data_info.iloc[:, 3].astype(int).tolist()

file_paths_exist = []
labels_exist = []

for row, label in zip(file_paths, labels):
    all_arrays = []
    for file_path in row:
        if os.path.exists(file_path):
            loaded_array = np.load(file_path).flatten()
            # Zero padding to match length
            if len(loaded_array) < 8480:
                padded_array = np.pad(loaded_array, (0, 8480 - len(loaded_array)), mode='constant')
                all_arrays.append(padded_array)
            else:
                all_arrays.append(loaded_array)
        else:
            print(f"File not found: {file_path}")
            break
    file_paths_exist.append(np.concatenate(all_arrays))
    labels_exist.append(label)

labels = np.array(labels_exist)
data = np.array(file_paths_exist)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

report = poly(data, labels, n_folds=8, scale=True, exclude=['Decision Tree', 'Random Forest'])

report.plot_scores(path='og_data')
