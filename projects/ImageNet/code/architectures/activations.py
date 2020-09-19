
""" Custom activation functions """

from tensorflow.keras.activations import tanh


def UV_activation(x):
    """ Activation function for predicting the U and V colour components - their range is [-0.5, 0.5] """
    return 0.5*tanh(x)


