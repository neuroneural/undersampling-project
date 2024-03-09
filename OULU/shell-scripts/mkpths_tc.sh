#!/bin/bash

SUBJECTS_FILE="/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt"
IFS=$'\n' subjects=(`cat $SUBJECTS_FILE`)
DATADIR=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU

PATHS_FILE="/data/users2/jwardell1/undersampling-project/OULU/txt-files/tc_data.txt"
> $PATHS_FILE

n_sections=18

#20150210_tr100_section14.npy

for subject in "${subjects[@]}"
do
	for ((i = 0; i < $n_sections; i++)); do
		for tr in 2150 100
		do
			echo "${DATADIR}/${subject}/processed/${subject}_tr${tr}_section${i}.npy" >> $PATHS_FILE
		done
	done
done
