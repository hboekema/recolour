
""" U-Net Fully Convolutional Neural Network """


from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Reshape, Flatten

from architectures.activations import UV_activation


def MLP(img_dim=(256,256)):
    """Simple MLP architecture for generating UV colour components given the Y (luminance) component

    Network input: grayscale image of specified dimensions
    Network output: YUV image of specified dimensions

    Parameters
    ----------
    img_dim : tuple
        Dimensions of input image (including the Y channel)

    Returns
    -------
    generator_input : tf.keras.layers.Input
        Input image node
    generator_out : tf.keras.layers.Output
        Output image node
    """

    # Network input image
    generator_input = Input(shape=(*img_dim, 1), name="generator_input")

    # Flatten input
    flat_input = Flatten()(generator_input)

    # Network architecture
    generator_architecture = Dense(256, activation="relu")(flat_input)

    # Network output
    generator_output = Dense(img_dim[0] * img_dim[1] * 2, activation=UV_activation)(generator_architecture)
    generator_output = Reshape((*img_dim, 2))(generator_output)
    generator_output = Concatenate(axis=-1, name="generator_output_YUV")([generator_input, generator_output])

    return generator_input, generator_output


