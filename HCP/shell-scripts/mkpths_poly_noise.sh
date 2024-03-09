#!/bin/bash

PROJECT_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/hcp-project

PATH_FILE=/data/users2/jwardell1/undersampling-project/HCP/csv-files/data_poly_noise.csv
SUBJECTS_FILE=/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/subjects.txt

IFS=$'\n' subjects=($(cat $SUBJECTS_FILE))

> $PATH_FILE

for subject in "${subjects[@]}"; do
	for label in noise non-noise
	do
		if [[ "$label" == "noise" ]]; then
			sr1_path=${PROJECT_DIRECTORY}/HCP/${subject}/processed/fnc1_triu.npy
			sr2_path=${PROJECT_DIRECTORY}/HCP/${subject}/processed/fnc2_triu.npy
			concat_path=${PROJECT_DIRECTORY}/HCP/${subject}/processed/fnc_concat.npy
			echo "${sr1_path},${sr2_path},${concat_path},1" >> $PATH_FILE
		else
			sr1_path=${PROJECT_DIRECTORY}/HCP/${subject}/processed/FNC_${subject}.npy
			sr2_path=${PROJECT_DIRECTORY}/HCP/${subject}/processed/FNC_${subject}_ds_3.npy
			concat_path=${PROJECT_DIRECTORY}/HCP/${subject}/processed/f1f2concat.npy
			echo "${sr1_path},${sr2_path},${concat_path},0" >> $PATH_FILE
		fi
	done
done
