#!/bin/bash

PATHS_FILE=/data/users2/jwardell1/undersampling-project/OULU/txt-files/paths_cobrecov
COV_PATH=/data/users2/jwardell1/nshor_docker/examples/cobre-project/COV



> $PATHS_FILE

for SNR in 1.5 1.6 1.7 1.8 1.9 2.0
do
	for subID in 0 1 2 3
	do
		echo "$SNR" >> $PATHS_FILE
		echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/g0.pkl" >> $PATHS_FILE
		echo "$subID" >> $PATHS_FILE
		echo "${COV_PATH}/${subID}_cov.npy" >> $PATHS_FILE
		echo "${COV_PATH}/${subID}_chol.npy" >> $PATHS_FILE
	done
done
