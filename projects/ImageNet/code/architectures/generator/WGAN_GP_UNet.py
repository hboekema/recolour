
""" WGAN-applicable Fully Convolutional Neural Network """

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, UpSampling2D, AveragePooling2D

from architectures.activations import UV_activation
from architectures.blocks import UNetEncoderBlock, UNetDecoderBlock


def WGAN_GP_UNet(img_dim=(256,256), dropout_rate=0.5, relu_alpha=0.2):
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
    generator_E1 = UNetEncoderBlock(64, (3,3), strides=2, padding="same", activation="leaky_relu")(generator_input)
    #generator_E2 = UNetEncoderBlock(128, (3,3), strides=2, padding="same", activation="leaky_relu")(generator_E1)
    #generator_E3 = UNetEncoderBlock(256, (3,3), strides=2, padding="same", activation="leaky_relu")(generator_E2)
    #generator_E4 = UNetEncoderBlock(512, (3,3), strides=2, padding="same", activation="leaky_relu")(generator_E3)
    
    # Bottleneck layer
    generator_BN = Conv2D(512, (3,3), strides=1, padding="same")(generator_E1)
    #generator_BN = AveragePooling2D(pool_size=(2,2))(generator_BN)
    generator_BN = LeakyReLU(relu_alpha)(generator_BN)

    # Decoder
    #generator_D4 = UNetDecoderBlock(512, (3,3), strides=2, padding="same", activation="relu", dropout_rate=dropout_rate)(generator_BN, generator_E4)
    #generator_D3 = UNetDecoderBlock(512, (3,3), strides=2, padding="same", activation="leaky_relu", dropout_rate=dropout_rate)(generator_BN, generator_E3)
    #generator_D2 = UNetDecoderBlock(512, (3,3), strides=2, padding="same", activation="leaky_relu", dropout_rate=0.)(generator_D3, generator_E2)
    generator_D1 = UNetDecoderBlock(64, (3,3), strides=2, padding="same", activation="leaky_relu", dropout_rate=0.)(generator_BN, generator_E1)

    # Output layer - output channels are U and V components, so need to append to Y channel (input image)
    #generator_output = Conv2DTranspose(2, (3,3), strides=2, activation=UV_activation, padding="same", name="generator_output_UV")(generator_D1) 
    generator_output = Conv2D(2, (1,1), padding="same", activation=UV_activation)(generator_D1)
    generator_output = Concatenate(axis=-1, dtype=tf.dtypes.float32, name="generator_output_YUV")([generator_input, generator_output])

    return generator_input, generator_output

