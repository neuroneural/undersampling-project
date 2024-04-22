#!/bin/bash

paths_file=/data/users2/jwardell1/undersampling-project/HCP/txt-files/paths_graphs
project_dir=/data/users2/jwardell1/undersampling-project/HCP/shell-scripts

num_lines=`wc -l <  $paths_file`
num_total_runs=$(( $num_lines / 3 ))


startix=0
endix=$(( $num_total_runs - 1 ))
batch_size=1000

#sbatch --array=${startix}-${endix}%${batch_size} ${project_dir}/poly_noise.job
sbatch --array=${startix}-${endix} ${project_dir}/poly_noise.job
