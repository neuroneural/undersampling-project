#!/bin/bash

PATHSFILE=/data/users2/jwardell1/undersampling-project/HCP/txt-files/tc_data.txt
> $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/subjects.txt

while IFS= read -r subjectID; do
	echo "/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/${subjectID}/processed/TCOutMax_${subjectID}.mat" >> $PATHSFILE
done < $SUBJECTSFILE
