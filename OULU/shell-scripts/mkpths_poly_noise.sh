#!/bin/bash

PROJECT_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/oulu-project

#PATH_FILE=/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly.csv
PATH_FILE1=/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly_noise1.csv
PATH_FILE2=/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly_noise2.csv
PATH_FILE3=/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly_noise3.csv
SUBJECTS_FILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt

IFS=$'\n' subjects=($(cat $SUBJECTS_FILE))

> $PATH_FILE1
> $PATH_FILE2
> $PATH_FILE3

NUM_SECTIONS=80

for subject in "${subjects[@]}"; do
	for ((i = 0; i < $NUM_SECTIONS; i++)); do
		for label in noise non-noise
		do
			if [[ "$label" == "noise" ]]; then
				sr1_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr2150_section${i}_fnc1_triu_noise.npy
				sr2_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr100_section${i}_fnc2_triu_noise.npy
				concat_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_concat_section${i}_fnc_noise.npy
				#echo "${sr1_path},${sr2_path},${concat_path},1" >> $PATH_FILE
				echo "${sr1_path},1" >> $PATH_FILE1
				echo "${sr2_path},1" >> $PATH_FILE2
				echo "${concat_path},1" >> $PATH_FILE3
			else
				sr1_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr2150_section${i}_triu_fnc.npy
				sr2_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_tr100_section${i}_triu_fnc.npy
				concat_path=${PROJECT_DIRECTORY}/OULU/${subject}/processed/${subject}_concat_section${i}_fnc.npy
				#echo "${sr1_path},${sr2_path},${concat_path},0" >> $PATH_FILE
				echo "${sr1_path},0" >> $PATH_FILE1
				echo "${sr2_path},0" >> $PATH_FILE2
				echo "${concat_path},0" >> $PATH_FILE3
			fi
		done
	done
done
