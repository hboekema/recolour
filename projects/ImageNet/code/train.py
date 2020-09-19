
""" Main training file. Read in the configurations for the training run and execute the training loop. """


import os
import argparse
import yaml
from datetime import datetime
from tools.training import training_procedure

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

# Read in the configurations
if args.config is not None:
    config_path = args.config
else:
    config_path = "config/config.yaml"

with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
        #print(setup_params)
    except yaml.YAMLError as exc:
        print(exc)

# Set the ID of this training pass
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date, time and model architecture as a run-id
    run_id = datetime.now().strftime("{}_%Y-%m-%d_%H:%M:%S".format(setup_params["MODEL"]["ARCHITECTURE"]))

import tensorflow as tf
import tensorflow.keras as keras
print("Keras version: " + str(keras.__version__))
print("TF version: " + str(tf.__version__))


if __name__ == "__main__":
    # Select GPU/CPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = setup_params["GENERAL"]["GPU_ID"]
    if setup_params["GENERAL"]["GPU_ID"] != "-1":
        print("GPU used:" + str(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        print("CPU used")

    # Execute training loop
    training_procedure(setup_params, run_id)

