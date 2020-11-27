

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Concatenate


def LucyMLP(img_dim=[256, 256]):
    gen_input = Input(shape=[img_dim[0], img_dim[1], 1])
    gen_flattened = Flatten()(gen_input)
    gen_architecture = Dense(256, activation="relu", name="MLP1")(gen_flattened)
    gen_output = Dense(2*img_dim[0]*img_dim[1], activation="linear")(gen_architecture)
    gen_reshaped = Reshape(target_shape=[img_dim[0], img_dim[1], 2])(gen_output)
    gen_yuv = Concatenate(axis=-1)([gen_input, gen_reshaped])

    return gen_input, gen_yuv

