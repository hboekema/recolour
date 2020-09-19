
""" U-Net Fully Convolutional Neural Network """


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout

from architectures.activations import UV_activation
from architectures.layers import SymmetricPadding


def FCNN(img_dim=(256,256)):
    """U-Net FCNN architecture for generating UV colour components given the Y (luminance) component

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

    # Network architecture
    # Encoder
    # Encoder block 1
    generator_E1 = Conv2D(32, (3,3), activation="relu")(generator_input)
    generator_E1 = SymmetricPadding(padding=(1,1))(generator_E1)
    generator_E1 = Conv2D(32, (3,3), activation="relu")(generator_E1)
    generator_E1 = SymmetricPadding(padding=(1,1))(generator_E1)
    generator_E1D = MaxPooling2D((2,2))(generator_E1)

    # Encoder block 2
    generator_E2 = Conv2D(64, (3,3), activation="relu")(generator_E1D)
    generator_E2 = SymmetricPadding(padding=(1,1))(generator_E2)
    generator_E2D = MaxPooling2D((2,2))(generator_E2)

    # Encoder block 3
    #generator_E3 = Conv2D(128, (3,3), activation="relu")(generator_E2D)
    #generator_E3 = SymmetricPadding(padding=(1,1))(generator_E3)
    #generator_E3D = MaxPooling2D((2,2))(generator_E3)

    # Bottleneck
    #generator_BN = Conv2D(256, (3,3), activation="relu")(generator_E3D)
    generator_BN = Conv2D(256, (3,3), activation="relu")(generator_E2D)
    generator_BN = SymmetricPadding(padding=(1,1))(generator_BN)

    # Decoder
    # Decoder block 3
    #generator_D3 = UpSampling2D((2,2))(generator_BN)
    #generator_D3 = Conv2D(128, (3,3), activation="relu")(generator_D3)
    #generator_D3 = SymmetricPadding(padding=(1,1))(generator_D3)

    # Decoder block 2
    #generator_D2 = UpSampling2D((2,2))(generator_D3)
    generator_D2 = UpSampling2D((2,2))(generator_BN)
    generator_D2 = Conv2D(64, (3,3), activation="relu")(generator_D2)
    generator_D2 = SymmetricPadding(padding=(1,1))(generator_D2)

    # Decoder block 1
    generator_D1 = UpSampling2D((2,2))(generator_D2)
    generator_D1 = Conv2D(32, (3,3), activation="relu")(generator_D1)
    generator_D1 = SymmetricPadding(padding=(1,1))(generator_D1)
    generator_D1 = Conv2D(32, (3,3), activation="relu")(generator_D1)
    generator_D1 = SymmetricPadding(padding=(1,1))(generator_D1)

    # Output layer - output channels are U and V components, so need to append to Y channel (input image)
    generator_output = Conv2D(2, (1,1), activation=UV_activation, padding="same", name="generator_output_UV")(generator_D1)
    generator_output = Concatenate(axis=-1, name="generator_output_YUV")([generator_input, generator_output])

    return generator_input, generator_output


