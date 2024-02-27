#!/bin/bash

PATHSFILE=sub_out_dirs.txt
> $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/subjects.txt

while IFS= read -r subjectID; do
	echo "/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/${subjectID}/processed" >> $PATHSFILE

done < $SUBJECTSFILE
