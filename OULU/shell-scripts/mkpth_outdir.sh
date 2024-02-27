#!/bin/bash

PATHSFILE="/data/users2/jwardell1/undersampling-project/OULU/txt-files/sub_out_dirs.txt"
> $PATHSFILE

SUBJECTSFILE=/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/subjects.txt

while IFS= read -r subjectID; do
	echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/${subjectID}/processed" >> $PATHSFILE
done < $SUBJECTSFILE
