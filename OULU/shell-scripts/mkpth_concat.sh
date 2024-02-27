#!/bin/bash

PATHSFILE=/data/users2/jwardell1/undersampling-project/OULU/txt-files/data.txt
> $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt


# FNC_20150210_TR100.npy
while IFS= read -r subjectID; do
	for tr in 2150 100
	do
		echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${subjectID}/processed/FNC_${subjectID}_TR${tr}.npy" >> $PATHSFILE
	done
done < $SUBJECTSFILE
