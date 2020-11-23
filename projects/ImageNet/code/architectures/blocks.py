

import tensorflow as tf
import tensorflow_addons as tfa

from abc import ABC, abstractmethod
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Concatenate, Lambda, UpSampling2D, AveragePooling2D

from .layers import SymmetricPadding
from .tools import padding_kernel_from_conv_kernel_size


class ConvBlock(ABC):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
            activation=None, dropout_rate=0., batchnorm=True,
            instancenorm=False, relu_alpha=0.2, training=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.instancenorm = instancenorm
        self.relu_alpha = relu_alpha
        self.training = training

        self.advanced_activations = ["leaky_relu"]

    def _correct_padding(self, input_):
        if self.padding == "symmetric":
            padding_kernel = padding_kernel_from_conv_kernel_size(self.kernel_size)
            input_ = SymmetricPadding(padding=padding_kernel)(input_)
            padding = "valid"
        else:
            padding = self.padding

        return input_, padding

    def _check_activation(self):
        if self.activation in self.advanced_activations:
            activation = None
        else:
            activation = self.activation

        return activation

    def _apply_activation(self, input_layer):
        if self.activation == "leaky_relu":
            input_layer = LeakyReLU(self.relu_alpha)(input_layer)
        
        return input_layer

    def _apply_instancenorm(self, input_layer):
        if self.instancenorm:
            output_layer = Lambda(lambda x: tfa.layers.InstanceNormalization()(x, training=self.training))(input_layer)
            return output_layer
        else:
            return input_layer
    
    def _apply_batchnorm(self, input_layer):
        if self.batchnorm:
            output_layer = BatchNormalization()(input_layer, training=self.training)
            return output_layer
        else:
            return input_layer

    def _apply_dropout(self, input_layer):
        if self.dropout_rate is not None and self.dropout_rate > 0.:
            output_layer = Dropout(self.dropout_rate)(input_layer, training=self.training)
            return output_layer
        else:
            return input_layer

    @abstractmethod
    def __call__(self, block_input):
        pass


class GeneratorBlock2D(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
            dropout_rate=0., batchnorm=True, relu_alpha=0.2):
        super(GeneratorBlock2D, self).__init__(filters, kernel_size,
                strides, padding, dropout_rate, batchnorm, relu_alpha)

    def __call__(self, block_input): 
        block_input, padding = self._correct_padding(block_input)
    
        block_architecture = Conv2D(self.filters, self.kernel_size, padding=padding)(block_input)
        block_architecture = LeakyReLU(self.relu_alpha)(block_architecture)
        block_architecture = self._apply_batchnorm(block_architecture)
        block_architecture = self._apply_dropout(block_architecture)
        
        return block_architecture


class PatchGANBlock(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="same", dropout_rate=0., relu_alpha=0.2):
        super(PatchGANBlock, self).__init__(filters, kernel_size,
                strides=strides, padding=padding, dropout_rate=dropout_rate,
                relu_alpha=relu_alpha)

    def __call__(self, block_input):
        block_input, padding = self._correct_padding(block_input)
        
        block_architecture = Conv2D(self.filters, self.kernel_size, strides=1, padding=padding)(block_input)
        block_architecture = LeakyReLU(self.relu_alpha)(block_architecture)
        block_architecture = AveragePooling2D(pool_size=self.strides)(block_architecture)
        block_architecture = self._apply_dropout(block_architecture)

        return block_architecture


# TODO: Change from batch norm to instance norm when done debugging
class UNetEncoderBlock(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="same",
            activation="leaky_relu", instancenorm=True):
        super(UNetEncoderBlock, self).__init__(filters, kernel_size,
                strides=strides, padding=padding, activation=activation,
                dropout_rate=0., batchnorm=instancenorm, instancenorm=False,
                relu_alpha=0., training=True)
    
    def __call__(self, block_input):
        block_input, padding = self._correct_padding(block_input)
        activation = self._check_activation()

        block_architecture = Conv2D(self.filters, self.kernel_size, strides=1, padding=padding, activation=activation)(block_input)
        #block_architecture = Conv2D(self.filters, self.kernel_size, strides=1, padding=padding, activation=activation)(block_architecture)
        block_architecture = AveragePooling2D(pool_size=self.strides)(block_architecture)
        block_architecture = self._apply_activation(block_architecture)
        block_architecture = self._apply_batchnorm(block_architecture)

        return block_architecture


class UNetDecoderBlock(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="same",
            activation="relu", dropout_rate=0.5):
        super(UNetDecoderBlock, self).__init__(filters, kernel_size,
                strides=strides, padding=padding, activation=activation,
                dropout_rate=dropout_rate, batchnorm=True, instancenorm=False,
                relu_alpha=0., training=True)
    
    def __call__(self, block_input, skip_input):
        block_input, padding = self._correct_padding(block_input)
        activation = self._check_activation()

        #block_architecture = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides, padding=padding, activation=activation)(block_input)
        block_architecture = UpSampling2D(size=self.strides)(block_input)
        block_architecture = Concatenate(axis=-1)([block_architecture, skip_input]) 
        block_architecture = Conv2D(self.filters, self.kernel_size, strides=1, padding=padding, activation=activation)(block_architecture)
        #block_architecture = Conv2D(self.filters, self.kernel_size, strides=1, padding=padding, activation=activation)(block_architecture)
        block_architecture = self._apply_activation(block_architecture)
        block_architecture = self._apply_batchnorm(block_architecture)
        block_architecture = self._apply_dropout(block_architecture)

        return block_architecture


class GeneratorBlock2D(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
            dropout_rate=0., batchnorm=True, relu_alpha=0.2):
        super(GeneratorBlock2D, self).__init__(filters, kernel_size,
                strides=strides, padding=padding, dropout_rate=dropout_rate,
                batchnorm=batchnorm, relu_alpha=relu_alpha)

    def __call__(self, block_input): 
        block_input, padding = self._correct_padding(block_input)
    
        block_architecture = Conv2D(self.filters, self.kernel_size, padding=padding)(block_input)
        block_architecture = LeakyReLU(self.relu_alpha)(block_architecture)
        block_architecture = self._apply_batchnorm(block_architecture)
        block_architecture = self._apply_dropout(block_architecture)

        return block_architecture


class DiscriminatorBlock2D(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
            dropout_rate=0., relu_alpha=0.2):
        super(DiscriminatorBlock2D, self).__init__(filters, kernel_size,
                strides=strides, padding=padding, dropout_rate=dropout_rate,
                relu_alpha=relu_alpha)

    def __call__(self, block_input):
        block_input, padding = self._correct_padding(block_input)
        
        block_architecture = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=padding)(block_input)
        block_architecture = LeakyReLU(self.relu_alpha)(block_architecture)
        block_architecture = self._apply_dropout(block_architecture)

        return block_architecture

