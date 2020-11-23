
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
    MODE = setup_params["GENERAL"]["MODE"]
    if MODE == "GAN":
        run_id = datetime.now().strftime("GAN_{}_{}_%Y-%m-%d_%H:%M:%S".format(setup_params["GENERATOR"]["ARCHITECTURE"], setup_params["DISCRIMINATOR"]["ARCHITECTURE"])) 
    else:
        run_id = datetime.now().strftime("{}_%Y-%m-%d_%H:%M:%S".format(setup_params["MODEL"]["ARCHITECTURE"]))

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
print("Keras version: " + str(keras.__version__))
print("TF version: " + str(tf.__version__))


if __name__ == "__main__":
    # Select GPU/CPU training
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        print("Memory growth allowed: True")

        # Enable mixed-precision training
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("Policy name: %s" % policy.name)
        print("Compute dtype %s" % policy.compute_dtype)
        print("Variable dtype: %s" % policy.variable_dtype)    
    else:
        print("CPU used")

    # Execute training loop
    training_procedure(setup_params, run_id)

