import pickle
from gunfolds.utils import graphkit as gk

with open('/data/users2/jwardell1/undersampling-project/HCP/txt-files/sub_out_dirs.txt', 'r') as file:
        lines = file.readlines()


num_graphs = 100

for i in range(len(lines)):
    for j in range(num_graphs):
        g = gk.ringmore(53, 10) 
        sub_out_dir = lines[i].strip()
        with open(f'{sub_out_dir}/g{j}.pkl', 'wb') as f:
             pickle.dump(g, f)