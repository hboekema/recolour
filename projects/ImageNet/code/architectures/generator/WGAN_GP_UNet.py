
""" WGAN-applicable Fully Convolutional Neural Network """

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, AveragePooling2D, UpSampling2D, Concatenate, Dropout, LeakyReLU, BatchNormalization

from architectures.activations import UV_activation
from architectures.layers import SymmetricPadding
from architectures.blocks import GeneratorBlock2D


def WGAN_GP_UNet(img_dim=(256,256), dropout_rate=0., batchnorm=True):
    """WGAN FCNN architecture for generating UV colour components given the Y (luminance) component

    Network input: grayscale image of specified dimensions
    Network output: YUV image of specified dimensions

    Parameters
    ----------
    img_dim : tuple
        Dimensions of input image (excluding the channel)

    Returns
    -------
    generator_input : tf.keras.layers.Input
        Input image node
    generator_out : tf.keras.layers.Output
        Output image node
    """

    # Network input image
    generator_input = Input(shape=(*img_dim, 1), name="generator_input")

    # Network architecture
    # Encoder
    # Encoder block 1
    generator_E1a = GeneratorBlock2D(32, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_input)
    generator_E1b = GeneratorBlock2D(32, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_E1a)
    #generator_E1D = Conv2D(32, (4,4), strides=2, padding="same")(generator_E1b)
    generator_E1D = AveragePooling2D((2,2))(generator_E1b)

    # Encoder block 2
    generator_E2 = GeneratorBlock2D(64, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_E1D)
    #generator_E2D = Conv2D(64, (4,4), strides=2, padding="same")(generator_E2)
    generator_E2D = AveragePooling2D((2,2))(generator_E2)

    # Bottleneck
    generator_BN = GeneratorBlock2D(128, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_E2D)

    # Decoder
    # Decoder block 2
    #generator_D2U = Conv2DTranspose(128, (4,4), strides=2, padding="same")(generator_BN)
    generator_D2U = UpSampling2D((2,2))(generator_BN)
    generator_D2 = Concatenate(axis=-1)([generator_D2U, generator_E2])
    generator_D2 = GeneratorBlock2D(64, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_D2)

    # Decoder block 1
    #generator_D1U = Conv2DTranspose(64, (4,4), strides=2, padding="same")(generator_D2)
    generator_D1U = UpSampling2D((2,2))(generator_D2)
    generator_D1 = Concatenate(axis=-1)([generator_D1U, generator_E1b])
    generator_D1b = GeneratorBlock2D(32, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_D1)
    generator_D1a = GeneratorBlock2D(32, (3,3), padding="symmetric", dropout_rate=dropout_rate, batchnorm=batchnorm)(generator_D1b)
    
    # Output layer - output channels are U and V components, so need to append to Y channel (input image)
    generator_output = Conv2D(2, (1,1), activation=UV_activation, padding="same", name="generator_output_UV")(generator_D1a)
    generator_output = Concatenate(axis=-1, dtype=tf.dtypes.float32, name="generator_output_YUV")([generator_input, generator_output])

    return generator_input, generator_output

