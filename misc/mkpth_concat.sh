#!/bin/bash

PATHSFILE=data.txt
touch $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt

while IFS= read -r subjectID; do
	for tr in 100 2150
	do
		echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${subjectID}/processed/FNC_${subjectID}_TR${tr}.npy" >> $PATHSFILE
	done

done < $SUBJECTSFILE
