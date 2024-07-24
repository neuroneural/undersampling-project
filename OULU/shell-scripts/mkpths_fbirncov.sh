#!/bin/bash

PATHS_FILE=/data/users2/jwardell1/undersampling-project/OULU/txt-files/paths_fbirncov
COV_PATH=/data/users2/jwardell1/nshor_docker/examples/fbirn-project/COV



> $PATHS_FILE

#for SNR in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5
for SNR in 2.6 2.7 2.8 2.9 3.0
do
	for subID in 000300655084 000907477089 001052733667 001374718419
	do
		echo "$SNR" >> $PATHS_FILE
		echo "/data/users2/jwardell1/nshor_docker/examples/oulu-project/OULU/g0.pkl" >> $PATHS_FILE
		echo "$subID" >> $PATHS_FILE
		echo "${COV_PATH}/${subID}_cov.npy" >> $PATHS_FILE
		echo "${COV_PATH}/${subID}_chol.npy" >> $PATHS_FILE
	done
done
