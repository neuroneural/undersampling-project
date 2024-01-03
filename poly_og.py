import numpy as np
import pandas as pd
from polyssifier import poly


data_info = pd.read_csv('data_poly.csv', header=None)
file_paths = data_info[0].tolist()

labels = data_info[1].astype(int).tolist()
labels = np.array(labels)

data = [np.load(file_path) for file_path in file_paths]
data = np.array(data)
data = data.reshape(data.shape[0], data.shape[1]*data.shape[1])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)


report = poly(data, labels, n_folds=8, scale=True, exclude=['Decision Tree','Random Forest'])

report.plot_scores(path='og_data')
