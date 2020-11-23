
import numpy as np


def padding_dim_for_kernel_dim(kernel_dim):
    assert kernel_dim > 0
    if kernel_dim % 2 == 1:
        # If odd
        padding_dim = int((kernel_dim - 1)/2)
    else:
        # If even
        padding_dim = int(kernel_dim/2)

    return padding_dim


def padding_kernel_from_conv_kernel_size(kernel_size):
    # Calculate the padding kernel size required to maintain the dimensions of an input to a convolutional layer using kernels with specified size
    assert np.all(np.array(kernel_size) > 0)
    
    padding_kernel = (padding_dim_for_kernel_dim(dim) for dim in kernel_size)
    return padding_kernel

