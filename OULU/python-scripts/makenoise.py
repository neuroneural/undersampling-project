import pickle
from gunfolds.utils import graphkit as gk

out_dir = '/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/'

num_graphs = 5

for j in range(num_graphs):
    g = gk.ringmore(53, 10) 
    with open(f'{out_dir}/g{j}.pkl', 'wb') as f:
            pickle.dump(g, f)



out_dir = '/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/'

num_graphs = 5

for j in range(num_graphs):
    g = gk.ringmore(53, 10) 
    with open(f'{out_dir}/g{j}.pkl', 'wb') as f:
            pickle.dump(g, f)