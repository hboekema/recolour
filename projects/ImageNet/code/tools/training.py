
""" File with helper functions for training the model """


import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from tools.models import get_architecture_inputs_outputs, get_architecture_loss
from tools.experiment_setup import setup_exp_directory
from tools.data_generator import ImageGenerator
from tools.callbacks import LoggingCallback
from tools.data import load_images, gather_images_in_dir
from tools.gan import GANModel

from architectures.losses import generator_loss, discriminator_loss, wasserstein_generator_loss, wasserstein_discriminator_loss


def setup_data_generator(DATA_PATH, BATCH_SIZE, IMG_DIM, SHUFFLE):
    """Setup a training data generator using the data in the directory specified by the provided path.

    Parameters
    ----------
    DATA_PATH : str
        Path to directory containing training images
    BATCH_SIZE : int
        Training batch size
    IMG_DIM : tuple
        Tuple of (width, height) representing the input image dimension of the network
    SHUFFLE : bool
        Flag for determining whether to shuffle data at end of every epoch

    Returns
    -------
    tf.keras.utils.Sequence
        The image data generator
    """
    data_gen = ImageGenerator(DATA_PATH, batch_size=BATCH_SIZE, img_dim=IMG_DIM, shuffle=SHUFFLE)
    return data_gen


def setup_callbacks(EXP_DIR, DATA, PREDICTION_PERIOD, MODEL_SAVE_PERIOD, VAL_DATA=None):
    """Setup training callbacks for managing and inspecting the network's internal state during training

    """
    logging_cb = LoggingCallback(EXP_DIR, DATA, period=PREDICTION_PERIOD, mode="train", show=False)
    model_save_cb = ModelCheckpoint(os.path.join(EXP_DIR, "models",  "model.{epoch:02d}-{loss:.4f}.hdf5"),
        monitor='loss', verbose=1, save_best_only=False, mode='auto',
        save_freq=MODEL_SAVE_PERIOD, save_weights_only=True)

    # TODO: add model_save_cb back in when done debugging
    #callbacks = [logging_cb, model_save_cb]
    callbacks = [logging_cb]

    if VAL_DATA is not None:
        val_logging_cb = LoggingCallback(EXP_DIR, VAL_DATA, period=PREDICTION_PERIOD, mode="val", show=False)
        callbacks.append(val_logging_cb)

    return callbacks


def setup_train_model(ARCHITECTURE, PARAMS, LEARNING_RATE=0.001):
    """Retrieve architecture, setup optimizer, define losses and their weights, and compile the Keras Model

    Parameters
    ----------
    ARCHITECTURE : str
        Architecture name
    PARAMS: dict
        Parameters required by chosen architecture
    LEARNING_RATE : float
        Optimizer learning rate

    Returns
    -------
    tf.keras.models.Model
        Compiled Keras model
    """

    # Retrieve architecture input and outputs
    model_inputs, model_outputs = get_architecture_inputs_outputs(ARCHITECTURE, PARAMS)

    # Setup optimizer
    optimizer = Adam(lr=LEARNING_RATE)

    # Define losses and weights
    loss, loss_weights = get_architecture_loss(ARCHITECTURE)

    # Compile model
    model = Model(inputs=model_inputs, outputs=model_outputs)
    model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics={}
            )

    model.summary()

    return model


def setup_GAN_submodel(ARCHITECTURE, PARAMS, LEARNING_RATE, submodel_name):
    """Retrieve architecture, setup optimizer, and get losses for the generator or discriminator Model

    Parameters
    ----------
    ARCHITECTURE : str
        Architecture name
    PARAMS: dict
        Parameters required by chosen architecture
    LEARNING_RATE : float
        Optimizer learning rate
    submodel_name : str
        Name of the submodel (generator or discriminator) to create

    Returns
    -------
    tf.keras.models.Model
        Keras model
    """

    # Retrieve architecture input and outputs
    model_inputs, model_outputs = get_architecture_inputs_outputs(ARCHITECTURE, PARAMS)


    # Get loss
    if submodel_name == "generator":
        # Generator optimizer
        optimizer = Adam(lr=LEARNING_RATE)
        #loss_fn = generator_loss
        loss_fn = wasserstein_generator_loss
    elif submodel_name == "discriminator":
        # Discriminator optimizer - use RMSprop because momentum can mess with GAN convergence (see arXiv:1701.07875)
        optimizer = RMSprop(lr=LEARNING_RATE, momentum=0.0)
        #loss_fn = discriminator_loss
        loss_fn = wasserstein_discriminator_loss
    else:
        raise ValueError("'submodel_name' should be one of 'generator' or 'discriminator'. Value given: '{}'".format(submodel_name))

    # Create and return  model
    model = Model(inputs=model_inputs, outputs=model_outputs)
    return model, optimizer, loss_fn


def setup_GAN(GEN_ARCHITECTURE, DISC_ARCHITECTURE, GEN_PARAMS, DISC_PARAMS,
        GEN_LEARNING_RATE=0.001, DISC_LEARNING_RATE=0.001,
        GEN_RUNS_PER_EPOCH=1, DISC_RUNS_PER_EPOCH=5, APPLY_GP=False,
        GP_WEIGHT=10, APPLY_L1_PENALTY=False, L1_WEIGHT=1, REPLAY_BUFFER_EPOCHS=None):
    # Create the generator and discriminator
    generator, gen_optimizer, gen_loss = setup_GAN_submodel(GEN_ARCHITECTURE, GEN_PARAMS, GEN_LEARNING_RATE, "generator")
    discriminator, disc_optimizer, disc_loss = setup_GAN_submodel(DISC_ARCHITECTURE, DISC_PARAMS, DISC_LEARNING_RATE, "discriminator")

    # Create the GAN
    gan = GANModel(generator, discriminator, gen_loss, disc_loss,
            gen_optimizer, disc_optimizer, GEN_RUNS_PER_EPOCH,
            DISC_RUNS_PER_EPOCH, APPLY_GP, GP_WEIGHT, APPLY_L1_PENALTY, L1_WEIGHT, REPLAY_BUFFER_EPOCHS)
    return gan


def train_model_with_data_generator(model, data_generator, PARAMS, validation_data=None, callbacks=None):
    """Train a Model using Keras' fit method and a custom data generator.

    Parameters
    ----------
    model : tf.keras.models.Model
        The Keras Model instance to be trained
    data_generator : tf.keras.utils.Sequence
        Data generator to use in training
    PARAMS : dict
        Parameters to be fed to fit_generator method
    validation_data : tf.keras.utils.Sequence
        Validation data to use in training
    callbacks : list
        Optional callbacks to use in training

    Returns
    -------

    """

    # Train model
    history = model.fit(
            data_generator,
            callbacks=callbacks,
            validation_data=validation_data,
            **PARAMS
            )

    return history


def training_procedure(SETUP_PARAMS, RUN_ID):
    """ Train a network with the given parameters """
    # Get training mode
    MODE = SETUP_PARAMS["GENERAL"]["MODE"]
    print("Training mode: {}".format(MODE))

    # Experiment setup parameters
    WRITE_DIR = SETUP_PARAMS["SETUP"]["WRITE_DIR"]
    if WRITE_DIR is None:
        print("Write directory: default")
    else:
        print("Write directory: {}".format(WRITE_DIR))

    # Data parameters
    TRAIN_DATA_PATH = SETUP_PARAMS["DATA"]["TRAIN_DATA_PATH"]
    VAL_DATA_PATH = SETUP_PARAMS["DATA"]["VAL_DATA_PATH"]

    # Model parameters & training parameters
    IMG_DIM = SETUP_PARAMS["MODEL"]["IMG_DIM"]
    if MODE == "GAN":
        GEN_ARCHITECTURE = SETUP_PARAMS["GENERATOR"]["ARCHITECTURE"]
        GEN_LEARNING_RATE = SETUP_PARAMS["GENERATOR"]["GEN_LEARNING_RATE"]
        GEN_RUNS_PER_EPOCH = SETUP_PARAMS["GENERATOR"]["RUNS_PER_EPOCH"]
        APPLY_L1_PENALTY = SETUP_PARAMS["GENERATOR"]["APPLY_L1_PENALTY"]
        L1_WEIGHT = SETUP_PARAMS["GENERATOR"]["L1_WEIGHT"]
        GEN_PARAMS = SETUP_PARAMS["GEN_PARAMS"]

        DISC_ARCHITECTURE = SETUP_PARAMS["DISCRIMINATOR"]["ARCHITECTURE"]
        DISC_LEARNING_RATE = SETUP_PARAMS["DISCRIMINATOR"]["DISC_LEARNING_RATE"]
        DISC_RUNS_PER_EPOCH = SETUP_PARAMS["DISCRIMINATOR"]["RUNS_PER_EPOCH"]
        APPLY_GP = SETUP_PARAMS["DISCRIMINATOR"]["APPLY_GP"]
        GP_WEIGHT = SETUP_PARAMS["DISCRIMINATOR"]["GP_WEIGHT"]
        REPLAY_BUFFER_EPOCHS = SETUP_PARAMS["DISCRIMINATOR"]["REPLAY_BUFFER_EPOCHS"]
        DISC_PARAMS = SETUP_PARAMS["DISC_PARAMS"]
    else:
        ARCHITECTURE = SETUP_PARAMS["MODEL"]["ARCHITECTURE"]
        PARAMS = SETUP_PARAMS["PARAMS"]
        LEARNING_RATE = SETUP_PARAMS["TRAIN"]["LEARNING_RATE"]

    BATCH_SIZE = SETUP_PARAMS["TRAIN"]["BATCH_SIZE"]
    EPOCHS = SETUP_PARAMS["TRAIN"]["EPOCHS"]
    STEPS_PER_EPOCH = SETUP_PARAMS["TRAIN"]["STEPS_PER_EPOCH"]
    MODEL_SAVE_PERIOD = SETUP_PARAMS["TRAIN"]["MODEL_SAVE_PERIOD"]
    PREDICTION_PERIOD = SETUP_PARAMS["TRAIN"]["PREDICTION_PERIOD"]
    SHUFFLE = SETUP_PARAMS["TRAIN"]["SHUFFLE"]
    TRAIN_SAMPLES = SETUP_PARAMS["TRAIN"]["TRAIN_SAMPLES"]

    # Validation parameters
    VALIDATION_SAMPLES = SETUP_PARAMS["VAL"]["VALIDATION_SAMPLES"]

    # Visualisation parameters
    VIS_SAMPLES = SETUP_PARAMS["VIS"]["VIS_SAMPLES"]

    # Set the experiment directory up
    exp_dir, _, _, _, _ = setup_exp_directory(run_id=RUN_ID, write_dir=WRITE_DIR)

    # Set the data generator up
    TRAIN_IMAGE_PATHS = gather_images_in_dir(TRAIN_DATA_PATH)
    if TRAIN_SAMPLES is not None and TRAIN_SAMPLES > 0:
        assert TRAIN_SAMPLES < len(TRAIN_IMAGE_PATHS), "More training samples selected than in data"
        TRAIN_IMAGE_PATHS = TRAIN_IMAGE_PATHS[:TRAIN_SAMPLES]
    data_gen = setup_data_generator(TRAIN_IMAGE_PATHS, BATCH_SIZE, IMG_DIM, SHUFFLE)

    # Prepare validation data
    VAL_IMAGE_PATHS = gather_images_in_dir(VAL_DATA_PATH)
    assert VALIDATION_SAMPLES < len(VAL_IMAGE_PATHS), "More validation samples selected than in data"
    VAL_IMAGE_PATHS = VAL_IMAGE_PATHS[:VALIDATION_SAMPLES]
    val_data = load_images(VAL_IMAGE_PATHS, img_dim=IMG_DIM)

    # Prepare the visualisation data
    assert VIS_SAMPLES <= len(TRAIN_IMAGE_PATHS), "More visualisation samples selected than there are training samples"
    assert VIS_SAMPLES <= VALIDATION_SAMPLES, "More visualisation samples selected than there are validation samples"
    train_data_for_vis = load_images(TRAIN_IMAGE_PATHS[:VIS_SAMPLES], img_dim=IMG_DIM)
    val_data_for_vis = val_data[:VIS_SAMPLES]

    # Set the callbacks up
    callbacks = setup_callbacks(exp_dir, train_data_for_vis, PREDICTION_PERIOD,
            MODEL_SAVE_PERIOD*STEPS_PER_EPOCH, VAL_DATA=val_data_for_vis)


    if MODE == "GAN":
        # Create the model
        model = setup_GAN(GEN_ARCHITECTURE, DISC_ARCHITECTURE, GEN_PARAMS,
                DISC_PARAMS, GEN_LEARNING_RATE, DISC_LEARNING_RATE,
                GEN_RUNS_PER_EPOCH, DISC_RUNS_PER_EPOCH, APPLY_GP, GP_WEIGHT,
                APPLY_L1_PENALTY, L1_WEIGHT, REPLAY_BUFFER_EPOCHS)
    else:
        # Create the model
        model = setup_train_model(ARCHITECTURE, PARAMS, LEARNING_RATE)

    # Train model
    history = train_model_with_data_generator(model, data_gen, {"epochs": EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH, "batch_size": BATCH_SIZE}, validation_data=val_data, callbacks=callbacks)

