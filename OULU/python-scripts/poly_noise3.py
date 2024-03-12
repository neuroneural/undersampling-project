import numpy as np
import pandas as pd
from polyssifier import poly
import os

data_info = pd.read_csv('/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly_noise3.csv', header=None)
file_paths = data_info.iloc[:, 0].values.tolist()
labels = data_info.iloc[:, 1].astype(int).tolist()

file_paths_exist = []
labels_exist = []

for row, label in zip(file_paths, labels):
    all_arrays = []
    file_path = row
    if os.path.exists(file_path):
           
        loaded_array = np.load(file_path)
        
        file_paths_exist.append(loaded_array)
        labels_exist.append(label)
        print(f'len(loaded_array) - {len(loaded_array)}')
        
    else:
        print(f"File not found: {file_path}")
        break

    
labels = np.array(labels_exist)
data = np.array(file_paths_exist)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

report = poly(data, labels, n_folds=8, scale=True, exclude=['Decision Tree', 'Random Forest'])

report.plot_scores(path='noise_data3')

