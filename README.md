# Undersampling Anomaly Detection Project


## How to use



Create a conda environment using usp_environment.yml file and activate it.

```
conda env create -f usp_environment.yml
conda activate usp
```


You will need to set the python path in the following way:

```
PYTHONPATH=./undersampling-project/scripts/
```

To run an experiment for MLP, NB, and LR classifiers, run the `experiment.py` script. Here is its usage:


```
usage: experiment.py [-h] -n NOISE_DATASET -s SIGNAL_DATASET [-i SNR_INT [SNR_INT ...]] [-f N_FOLDS] [-nn NUM_NOISE] [-v]

options:
  -h, --help            show this help message and exit
  -n NOISE_DATASET, --noise-dataset NOISE_DATASET
                        noise dataset name (FBIRN, COBRE, VAR)
  -s SIGNAL_DATASET, --signal-dataset SIGNAL_DATASET
                        signal dataset name (OULU, HCP)
  -i SNR_INT [SNR_INT ...], --snr-int SNR_INT [SNR_INT ...]
                        upper, lower, step of SNR interval
  -f N_FOLDS, --n-folds N_FOLDS
                        number of folds for cross-validation
  -nn NUM_NOISE, --num_noise NUM_NOISE
                        number of noise iterations
  -v, --verbose         turn on debug logging
```



The SVM experiments need to use a different environment. Create the thundersvm environment from the thundersvm_environment.yml file then activate it. 

```
conda env create -f thundersvm_environment.yml
conda activate thundersvm
```

To run an SVM classification experiment, you will need the model weights. You can generate them on your own using a hyper parameter search, or they are available by request. 

To run a hyperparameter search, run the `hps_thundersvm.py` script. Here is its usage:

```
usage: hps_thundersvm.py [-h] -n NOISE_DATASET -s SIGNAL_DATASET -k {linear,rbf} [-i SNR_INT [SNR_INT ...]] [-f N_FOLDS] [-v VERBOSE]

options:
  -h, --help            show this help message and exit
  -n NOISE_DATASET, --noise-dataset NOISE_DATASET
                        noise dataset name
  -s SIGNAL_DATASET, --signal-dataset SIGNAL_DATASET
                        signal dataset name
  -k {linear,rbf}, --kernel-type {linear,rbf}
                        type of SVM kernel
  -i SNR_INT [SNR_INT ...], --snr-int SNR_INT [SNR_INT ...]
                        upper, lower, step of SNR interval
  -f N_FOLDS, --n-folds N_FOLDS
                        number of folds for cross-validation
  -v VERBOSE, --verbose VERBOSE
                        turn on debug logging
```


To run an SVM experiment, use the `nopoly_thundersvm.py` script. Here is its usage. 

```
usage: nopoly_thundersvm.py [-h] -n NOISE_DATASET -s SIGNAL_DATASET -k {linear,rbf} [-i SNR_INT [SNR_INT ...]] [-f N_FOLDS] [-nn NUM_NOISE] [-v VERBOSE]

options:
  -h, --help            show this help message and exit
  -n NOISE_DATASET, --noise-dataset NOISE_DATASET
                        noise dataset name
  -s SIGNAL_DATASET, --signal-dataset SIGNAL_DATASET
                        signal dataset name
  -k {linear,rbf}, --kernel-type {linear,rbf}
                        type of SVM kernel
  -i SNR_INT [SNR_INT ...], --snr-int SNR_INT [SNR_INT ...]
                        upper, lower, step of SNR interval
  -f N_FOLDS, --n-folds N_FOLDS
                        number of folds for cross-validation
  -nn NUM_NOISE, --num_noise NUM_NOISE
                        number of noise iterations
  -v VERBOSE, --verbose VERBOSE
                        turn on debug logging
```



For results, see the `plot.ipynb` notebook. You can modify this script to plot results of your classification experiments. 
