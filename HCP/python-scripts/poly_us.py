import numpy as np
import pandas as pd
from polyssifier import poly
import os

data_info = pd.read_csv('data_poly_us.csv', header=None)
file_paths = data_info[0].tolist()
labels = data_info[1].astype(int).tolist()

# Filter file_paths and labels simultaneously
file_paths_exist = []
labels_exist = []

for file_path, label in zip(file_paths, labels):
    if os.path.exists(file_path):
        file_paths_exist.append(file_path)
        labels_exist.append(label)

labels = np.array(labels_exist)

data = [np.load(file_path) for file_path in file_paths_exist]
data = np.array(data)
data = data.reshape(data.shape[0], data.shape[1] * data.shape[1])

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

report = poly(data, labels, n_folds=8, scale=True, exclude=['Decision Tree', 'Random Forest'])

report.plot_scores(path='us_data')

