
""" WGAN-applicable Fully Convolutional Neural Network """

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.constraints import MinMaxNorm

from architectures.activations import UV_activation
from architectures.layers import SymmetricPadding


def WGAN_UNet(img_dim=(256,256), weight_clip_norm=0.01):
    """WGAN FCNN architecture for generating UV colour components given the Y (luminance) component

    Network input: grayscale image of specified dimensions
    Network output: YUV image of specified dimensions

    Parameters
    ----------
    img_dim : tuple
        Dimensions of input image (excluding the channel)
    weight_clip_norm : int
        Absolute value of kernel weights to clip to

    Returns
    -------
    generator_input : tf.keras.layers.Input
        Input image node
    generator_out : tf.keras.layers.Output
        Output image node
    """

    # Clip the weights of the network to satisfy the Lipschitz conditions, allowing the discriminator to be trained to optimality
    min_max_norm = MinMaxNorm(min_value=-weight_clip_norm, max_value=weight_clip_norm)

    # Network input image
    generator_input = Input(shape=(*img_dim, 1), name="generator_input")

    # Network architecture
    # Encoder
    # Encoder block 1
    generator_E1 = Conv2D(32, (3,3), kernel_constraint=min_max_norm)(generator_input)
    generator_E1 = LeakyReLU()(generator_E1)
    generator_E1 = BatchNormalization()(generator_E1)
    generator_E1 = Dropout(0.1)(generator_E1)
    generator_E1 = SymmetricPadding(padding=(1,1))(generator_E1)
    
    generator_E1 = Conv2D(32, (3,3), kernel_constraint=min_max_norm)(generator_E1)
    generator_E1 = LeakyReLU()(generator_E1)
    generator_E1 = BatchNormalization()(generator_E1)
    generator_E1 = Dropout(0.1)(generator_E1)
    generator_E1 = SymmetricPadding(padding=(1,1))(generator_E1)
    
    generator_E1D = MaxPooling2D((2,2))(generator_E1)

    # Encoder block 2
    generator_E2 = Conv2D(64, (3,3), kernel_constraint=min_max_norm)(generator_E1D)
    generator_E2 = LeakyReLU()(generator_E2)
    generator_E2 = BatchNormalization()(generator_E2)
    generator_E2 = Dropout(0.1)(generator_E2)
    generator_E2 = SymmetricPadding(padding=(1,1))(generator_E2)
    
    generator_E2D = MaxPooling2D((2,2))(generator_E2)

    # Bottleneck
    generator_BN = Conv2D(256, (3,3), kernel_constraint=min_max_norm)(generator_E2D)
    generator_BN = LeakyReLU()(generator_BN)
    generator_BN = BatchNormalization()(generator_BN)
    generator_BN = Dropout(0.1)(generator_BN)
    generator_BN = SymmetricPadding(padding=(1,1))(generator_BN)

    # Decoder
    # Decoder block 2
    generator_D2U = UpSampling2D((2,2))(generator_BN)

    generator_D2 = Concatenate(axis=-1)([generator_D2U, generator_E2])
    generator_D2 = Conv2D(64, (3,3), kernel_constraint=min_max_norm)(generator_D2)
    generator_D2 = LeakyReLU()(generator_D2)
    generator_D2 = BatchNormalization()(generator_D2)
    generator_D2 = Dropout(0.1)(generator_D2)
    generator_D2 = SymmetricPadding(padding=(1,1))(generator_D2)

    # Decoder block 1
    generator_D1U = UpSampling2D((2,2))(generator_D2)
    
    generator_D1 = Concatenate(axis=-1)([generator_D1U, generator_E1])
    generator_D1 = Conv2D(32, (3,3), kernel_constraint=min_max_norm)(generator_D1)
    generator_D1 = LeakyReLU()(generator_D1)
    generator_D1 = BatchNormalization()(generator_D1)
    generator_D1 = Dropout(0.1)(generator_D1)
    generator_D1 = SymmetricPadding(padding=(1,1))(generator_D1)
    
    generator_D1 = Conv2D(32, (3,3), kernel_constraint=min_max_norm)(generator_D1)
    generator_D1 = LeakyReLU()(generator_D1)
    generator_D1 = BatchNormalization()(generator_D1)
    generator_D1 = Dropout(0.1)(generator_D1)
    generator_D1 = SymmetricPadding(padding=(1,1))(generator_D1)

    # Output layer - output channels are U and V components, so need to append to Y channel (input image)
    generator_output = Conv2D(2, (1,1), activation=UV_activation, padding="same", kernel_constraint=min_max_norm, name="generator_output_UV")(generator_D1)
    generator_output = Concatenate(axis=-1, dtype=tf.dtypes.float32, name="generator_output_YUV")([generator_input, generator_output])

    return generator_input, generator_output


