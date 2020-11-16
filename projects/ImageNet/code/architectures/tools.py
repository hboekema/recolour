
import numpy as np


def padding_kernel_from_conv_kernel_size(kernel_size):
    # Calculate the padding kernel size required to maintain the dimensions of an input to a convolutional layer using kernels with specified size
    # Only works for odd kernel sizes
    assert np.all(np.array(kernel_size) % 2 == 1)

    padding_kernel = (int((dim - 1)/2) for dim in kernel_size)
    return padding_kernel

