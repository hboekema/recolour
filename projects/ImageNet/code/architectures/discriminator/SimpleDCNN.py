
""" Simple DCNN discriminator network - outputs logits """


from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.constraints import MinMaxNorm



def SimpleDCNN(img_dim=(256,256), weight_clip_norm=0.01):
    """WGAN-suitable convolutional critic network for scoring RGB images.

    Network input: RGB image of specified dimensions
    Network output: image 'score', a real number in range (-inf, inf), with high-quality images scoring higher than low-quality images

    Parameters
    ----------
    img_dim : tuple
        Dimensions of input image (excluding the channel)
    weight_clip_norm : int
        Absolute value of kernel weights to clip to

    Returns
    -------
    disc_input : tf.keras.layers.Input
        Input (image) node
    disc_output : tf.keras.layers.Output
        Output node    
    """

    # Clip the weights of the network to satisfy the Lipschitz conditions, allowing the discriminator to be trained to optimality
    min_max_norm = MinMaxNorm(min_value=-weight_clip_norm, max_value=weight_clip_norm)
    
    # Input to the discriminator is a (w, h, 3) YUV image
    disc_input = Input(shape=(*img_dim, 3), name="discriminator_input")

    # Discriminator architecture
    disc_architecture = Conv2D(64, (5,5), padding="valid", kernel_constraint=min_max_norm)(disc_input)
    disc_architecture = LeakyReLU()(disc_architecture)
    disc_architecture = BatchNormalization()(disc_architecture)
    disc_architecture = MaxPooling2D()(disc_architecture)
    disc_architecture = Dropout(0.3)(disc_architecture)

    disc_architecture = Conv2D(128, (3,3), padding="valid", kernel_constraint=min_max_norm)(disc_input)
    disc_architecture = LeakyReLU()(disc_architecture)
    disc_architecture = BatchNormalization()(disc_architecture)
    disc_architecture = MaxPooling2D()(disc_architecture)
    disc_architecture = Dropout(0.3)(disc_architecture)

    disc_architecture = Conv2D(256, (3,3), padding="valid", kernel_constraint=min_max_norm)(disc_input)
    disc_architecture = LeakyReLU()(disc_architecture)
    disc_architecture = BatchNormalization()(disc_architecture)
    disc_architecture = MaxPooling2D()(disc_architecture)
    disc_architecture = Dropout(0.3)(disc_architecture)

    # Discriminator output (in logits)
    disc_architecture = Flatten()(disc_architecture)
    disc_output = Dense(1, kernel_constraint=min_max_norm, name="disc_output")(disc_architecture)

    return disc_input, disc_output

