#!/bin/bash

PROJECT_DIRECTORY=/data/users2/jwardell1/nshor_docker/examples/hcp-project

PATH_FILE=/data/users2/jwardell1/undersampling-project/HCP/data_poly_concat.csv
LABELS_FILE=/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/labels.txt

IFS=$'\n' entries=($(cat $LABELS_FILE))

> $PATH_FILE

for entry in "${entries[@]}"; do
    IFS=',' read -r subjectID label <<< "$entry"
    echo "/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/${subjectID}/processed/f1f2concat.npy,${label}" >> $PATH_FILE
done
