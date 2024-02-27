#!/bin/bash

PATHSFILE=/data/users2/jwardell1/undersampling-project/HCP/shell-scripts/data.txt
> $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/subjects.txt

while IFS= read -r subjectID; do
	for sr in 1 3
	do
		if [ "$sr" == 1  ]; then
			echo "/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/${subjectID}/processed/FNC_${subjectID}.npy" >> $PATHSFILE
		else
			echo "/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/${subjectID}/processed/FNC_${subjectID}_ds_3.npy" >> $PATHSFILE
		fi
	done

done < $SUBJECTSFILE
