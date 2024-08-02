#!/bin/bash

paths_file=/data/users2/jwardell1/undersampling-project/OULU/txt-files/paths_fbirncov

project_dir=/data/users2/jwardell1/undersampling-project/OULU/shell-scripts

num_lines=`wc -l <  $paths_file`
num_args=5
num_total_runs=$(( $num_lines / $num_args ))


startix=0
endix=$(( $num_total_runs - 1 ))
batch_size=12


watch "rm -rf /data/users2/jwardell1/tmp/pymp-*" & 

sbatch --array=${startix}-${endix}%${batch_size} ${project_dir}/poly_noise1.job
#sbatch --array=${startix}-${endix} ${project_dir}/poly_noise1.job
