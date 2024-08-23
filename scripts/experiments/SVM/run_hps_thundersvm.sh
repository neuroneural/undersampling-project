#!/bin/bash

CONFIG_FILE=$1

if [ -z $1 ]; then
	echo "please specify the config file"
	echo "usage: bash run_hps_thundersvm.sh <CONFIG FILENAME>"
	exit
fi


CONFIG_FILE=$1

# Parse the config file
while IFS=: read -r key value; do
    # Remove leading/trailing whitespace from key and value
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    
    # Store the values in variables
    case "$key" in
        signal_ds) SIGNAL_DS=$value ;;
        noise_df) NOISE_DF=$value ;;
        kernel) KERNEL=$value ;;
        noise_iter) NOISE_ITER=$value ;;
        snr_int) SNR_INT=$value ;;
        *) echo "Unknown key: $key" ;;
    esac
done < "$CONFIG_FILE"

ARGS="-s '$SIGNAL_DS' -n '$NOISE_DF' -k '$KERNEL' -nn $NOISE_ITER -i $SNR_INT"


# Name of the screen session
SCREEN_SESSION_NAME="hps_thundersvm_session_${SIGNAL_DS}"

# Path to the Python script
PYTHON_SCRIPT="hps_thundersvm.py"

# Log file path
LOG_FILE="hps_thundersvm.log"

# Start a screen session and run the Python script
screen -dmS $SCREEN_SESSION_NAME bash -c "
  echo 'Starting hps_thundersvm.py...' | tee -a $LOG_FILE
  python $PYTHON_SCRIPT $ARGS >> $LOG_FILE 2>&1
  if [ $? -eq 0 ]; then
    echo 'Process has completed successfully.' | tee -a $LOG_FILE
  else
    echo 'Process failed with an error.' | tee -a $LOG_FILE
  fi
  echo 'Log available at $LOG_FILE' | tee -a $LOG_FILE
  read -p 'Press Enter to exit...'
"

# Notify the user that the screen session has started
echo "Screen session $SCREEN_SESSION_NAME started."
echo "You can attach to it using: screen -r $SCREEN_SESSION_NAME"
echo "Log file: $LOG_FILE"

