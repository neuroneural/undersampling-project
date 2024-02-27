#!/bin/bash

PROJECT_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/oulu-project

PATH_FILE=/data/users2/jwardell1/undersampling-project/OULU/csv-files/data_poly_concat.csv

SUBJECTS_FILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt
IFS=$'\n' subjects=(`cat $SUBJECTS_FILE`)

> $PATH_FILE

for subjectID in "${subjects[@]}"; do
	for label in noise no-noise 
	do
		if [[  "$label" == "no-noise" ]]; then
    			echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${subjectID}/processed/f1f2concat.npy,0" >> $PATH_FILE
		else
    			echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${subjectID}/processed/f1f2concat_${label}.npy,1" >> $PATH_FILE
		fi
	done
done
