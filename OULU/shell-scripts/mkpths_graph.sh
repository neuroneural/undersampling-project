#!/bin/bash

PATHS_FILE=/data/users2/jwardell1/undersampling-project/OULU/txt-files/paths_graphs

> $PATHS_FILE

n_graphs=1
for SNR in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
	for ((i=0; i<$n_graphs; i++))
	do
		echo "$SNR" >> $PATHS_FILE
		echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/g${i}.pkl" >> $PATHS_FILE
		echo "${i}" >> $PATHS_FILE
	done
done
