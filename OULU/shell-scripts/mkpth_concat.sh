#!/bin/bash

PATHSFILE=/data/users2/jwardell1/undersampling-project/OULU/txt-files/data_concat.txt
> $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt


# FNC_20150210_TR100.npy
while IFS= read -r subjectID; do
	for (( i=0; i < 18; i++)); do
		for tr in 2150 100
		do
			#20150210_tr100_section11_triu_fnc.npy
			echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${subjectID}/processed/${subjectID}_tr${tr}_section${i}_triu_fnc.npy" >> $PATHSFILE
		done
	done
done < $SUBJECTSFILE
