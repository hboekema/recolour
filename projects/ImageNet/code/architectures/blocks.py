

import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, BatchNormalization

from .layers import SymmetricPadding
from .tools import padding_kernel_from_conv_kernel_size


class ConvBlock(ABC):
    def __init__(self, filters, kernel_size, strides=1, padding="valid", dropout_rate=0.,
            batchnorm=True, relu_alpha=0.2):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.relu_alpha = relu_alpha

    def _correct_padding(self, input_):
        if self.padding == "symmetric":
            padding_kernel = padding_kernel_from_conv_kernel_size(self.kernel_size)
            input_ = SymmetricPadding(padding=padding_kernel)(input_)
            padding = "valid"
        else:
            padding = self.padding

        return input_, padding

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
        if self.batchnorm:
            block_architecture = BatchNormalization()(block_architecture)
        block_output = Dropout(self.dropout_rate)(block_architecture)
        
        return block_output


class DiscriminatorBlock2D(ConvBlock):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
            dropout_rate=0., relu_alpha=0.2):
        super(DiscriminatorBlock2D, self).__init__(filters, kernel_size,
                strides, padding, dropout_rate, relu_alpha)

    def __call__(self, block_input):
        block_input, padding = self._correct_padding(block_input)
        
        block_architecture = Conv2D(self.filters, self.kernel_size,
                strides=self.strides, padding=padding)(block_input)
        block_architecture = LeakyReLU(self.relu_alpha)(block_architecture)
        block_output = Dropout(self.dropout_rate)(block_architecture)

        return block_output

