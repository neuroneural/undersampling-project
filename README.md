# Undersampling Anomaly Detection Project


## How to use



Create a conda environment using usp_environment.yml file and activate it.

```
conda env create -f usp_environment.yml
conda activate usp
```


You will need to set the python path in the following way:

```
PYTHONPATH=<FULL_PATH_TO_REPO>/undersampling-project/scripts/
```
where `FULL_PATH_TO_REPO` is the full path to the repository on your machine.

To run an experiment, run the `experiment.py` script. Here is its usage:


```
usage: experiment.py [-h] -n NOISE_DATASET -s SIGNAL_DATASET [-i SNR_INT [SNR_INT ...]] [-f N_FOLDS] [-nn NUM_NOISE] [-v] [-cv]

options:
  -h, --help            show this help message and exit
  -n NOISE_DATASET, --noise-dataset NOISE_DATASET                    [required]
                        noise dataset name (FBIRN, COBRE, VAR)
  -s SIGNAL_DATASET, --signal-dataset SIGNAL_DATASET                 [required]
                        signal dataset name (OULU, HCP)
  -i SNR_INT [SNR_INT ...], --snr-int SNR_INT [SNR_INT ...]
                        upper, lower, step of SNR interval
  -f N_FOLDS, --n-folds N_FOLDS
                        number of folds for cross-validation
  -nn NUM_NOISE, --num_noise NUM_NOISE
                        number of noise iterations
  -v, --verbose         turn on debug logging
  -cv, --cov-mat        use covariance matrix
```
For results, see the `plot.ipynb` notebook. You can modify this script to plot results of your classification experiments. 
