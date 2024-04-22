#!/bin/bash

PATHS_FILE=/data/users2/jwardell1/undersampling-project/OULU/txt-files/paths_graphs

> $PATHS_FILE


for SNR in 0.5 0.6 0.7 0.8 0.9 1.0 
do
	for ((i=0; i<5; i++))
	do
		echo "$SNR" >> $PATHS_FILE
		echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/g${i}.pkl" >> $PATHS_FILE
		echo "${i}" >> $PATHS_FILE
	done
done
