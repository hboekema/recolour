
""" Custom NN layers """


import tensorflow as tf
from tensorflow import keras


class SymmetricPadding(keras.layers.Layer):
    """ Pad Keras tensor using symmetric TF padding """
    def __init__(self, padding=(1,1), **kwargs):
        super(SymmetricPadding, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        paddings = [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0]]
        return tf.pad(input_tensor, paddings, mode='SYMMETRIC')

