#!/bin/bash

PATHS_FILE=/data/users2/jwardell1/undersampling-project/HCP/txt-files/paths_graphs

> $PATHS_FILE

n_graph=3
#for SNR in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
for SNR in 1.5 1.6 1.7 1.8 1.9 2.0
do
	for us_factor in 6 #2 3 4 5 6
	do
		for ((i=0; i<$n_graph; i++))
		do
			echo "$SNR" >> $PATHS_FILE
			echo "/data/users2/jwardell1/nshor_docker/examples/hcp-project/HCP/g${i}.pkl" >> $PATHS_FILE
			echo "${i}" >> $PATHS_FILE
			echo "${us_factor}" >> $PATHS_FILE
		done
	done
done
