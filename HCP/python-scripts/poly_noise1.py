import numpy as np
import pandas as pd
import logging  # Import the logging module
from polyssifier import poly_subject
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

data_info = pd.read_csv('/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly_noise1.csv', header=None)
file_paths = data_info.iloc[:, 0].values.tolist()
labels = data_info.iloc[:, 1].astype(int).tolist()

file_paths_exist = []
labels_exist = []
groups_exist = []  # To store inferred group labels

for row, label in zip(file_paths, labels):
    all_arrays = []
    file_path = row
    if os.path.exists(file_path):
        loaded_array = np.load(file_path)
        
        # Extract the group label from the immediate parent directory of the file path
        group_label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        
        file_paths_exist.append(loaded_array)
        labels_exist.append(label)
        groups_exist.append(group_label)
        logging.debug(f'len(loaded_array) - {len(loaded_array)}')  # Use logging.debug instead of print
        
    else:
        logging.error(f"File not found: {file_path}")  # Use logging.error for error messages
        continue

    
labels = np.array(labels_exist)
data = np.array(file_paths_exist)
groups = np.array(groups_exist)  # Use inferred group labels instead of subjects

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

report = poly_subject(data, labels, groups, n_folds=5, project_name='example', scale=True, exclude=['Decision Tree', 'Random Forest'],  scoring='f1')
print(report.scores)
