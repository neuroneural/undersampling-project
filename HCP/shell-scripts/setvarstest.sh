#!/bin/bash
SLURM_ARRAY_TASK_ID=0

export PATH=/data/users2/jwardell1/miniconda3/bin:$PATH

source /data/users2/jwardell1/miniconda3/etc/profile.d/conda.sh

CONDA_PATH=`which conda`

eval "$(${CONDA_PATH} shell.bash hook)"
conda activate /data/users2/jwardell1/miniconda3/envs/usp


export PATHS_FILE=/data/users2/jwardell1/undersampling-project/HCP/txt-files/paths_graphs
export paths_array=($(cat ${PATHS_FILE}))

export snr_ix=$(( 3*$SLURM_ARRAY_TASK_ID ))
export snr=${paths_array[${snr_ix}]}

export graph_ix=$(( 3*$SLURM_ARRAY_TASK_ID + 1 ))
export graph_filepath=${paths_array[${graph_ix}]}


export graph_no=$(( 3*$SLURM_ARRAY_TASK_ID + 2 ))
export graph_label=${paths_array[${graph_no}]}

echo "snr: ${snr}"
echo "graph_filepath: ${graph_filepath}"
echo "graph_label: ${graph_label}"


#python /data/users2/jwardell1/undersampling-project/HCP/python-scripts/poly_noise1.py $snr $graph_filepath $graph_no
