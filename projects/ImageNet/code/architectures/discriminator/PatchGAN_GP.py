
""" PatchGAN discriminator network for GP training - outputs logits """

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, Lambda

from architectures.blocks import PatchGANBlock


def PatchGAN_GP(img_dim=(256,256), dropout_rate=0.2, relu_alpha=0.2):
    """WGAN-suitable convolutional critic network for scoring RGB images.

    Network input: 3-channel colour image of specified dimensions
    Network output: image 'score', a real number in range (-inf, inf), with high-quality images scoring higher than low-quality images

    Parameters
    ----------
    img_dim : tuple
        Dimensions of input image (excluding the channel)
    relu_alpha : float
        Slope of negative values to pass through Leaky ReLu activations

    Returns
    -------
    disc_input : tf.keras.layers.Input
        Input (image) node
    disc_output : tf.keras.layers.Output
        Output node    
    """

    # Input to the discriminator is a (w, h, 3) YUV image
    disc_input = Input(shape=(*img_dim, 3), name="discriminator_input")

    # Discriminator architecture
    disc_architecture = PatchGANBlock(64, (3,3), strides=2, padding="same", dropout_rate=dropout_rate, relu_alpha=relu_alpha)(disc_input)
    disc_architecture = PatchGANBlock(128, (3,3), strides=2, padding="same", dropout_rate=dropout_rate, relu_alpha=relu_alpha)(disc_architecture)
    #disc_architecture = PatchGANBlock(256, (3,3), strides=2, padding="same", dropout_rate=dropout_rate, relu_alpha=relu_alpha)(disc_architecture)
    #disc_architecture = PatchGANBlock(512, (3,3), strides=2, padding="same", dropout_rate=dropout_rate, relu_alpha=relu_alpha)(disc_architecture)

    # Discriminator output (in logits)
    disc_output = Conv2D(1, (3,3), padding="valid")(disc_architecture)
    disc_output = GlobalAveragePooling2D()(disc_output)
    disc_output = Flatten(dtype=tf.dtypes.float32, name="disc_output")(disc_output)

    return disc_input, disc_output

