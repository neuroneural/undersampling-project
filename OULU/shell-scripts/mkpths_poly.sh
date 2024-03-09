#!/bin/bash

PROJECT_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/oulu-project

PATH_FILE=/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly.csv
SUBJECTS_FILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt

IFS=$'\n' subjects=($(cat $SUBJECTS_FILE))

> $PATH_FILE

NUM_SECTIONS=18

for subject in "${subjects[@]}"; do
	for ((i = 0; i < $NUM_SECTIONS; i++)); do
		for label in noise non-noise
		do
			if [[ "$label" == "noise" ]]; then
				sr1_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr2150_section${i}_fnc1_triu_noise.npy
				sr2_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr100_section${i}_fnc2_triu_noise.npy
				concat_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_concat_section${i}_fnc_noise.npy
				echo "${sr1_path},${sr2_path},${concat_path},1" >> $PATH_FILE
			else
				sr1_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr2150_section${i}_triu_fnc.npy
				sr2_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr100_section${i}_triu_fnc.npy
				concat_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_concat_section${i}_fnc.npy
				echo "${sr1_path},${sr2_path},${concat_path},0" >> $PATH_FILE
			fi
		done
	done
done
