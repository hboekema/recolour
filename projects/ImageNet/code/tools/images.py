""" Tools for processing images  """

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage import color


def write_to_img(img, text):
    # Text parameters
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,240)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2
    
    # Write text to img
    img = cv2.putText(img, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    return img


def reshape_image(img, img_dim):
    return np.array(cv2.resize(img, img_dim, interpolation=cv2.INTER_CUBIC))


def preprocess_img(img, img_dim=None):
    """Pre-process an image for deep network.

    Parameters
    ----------
    img : NumPy array
        RGB image of dimensions (n,m,3).
    Returns
    -------
    NumPy array
        Normalised luminance image of dimensions (h,w,1).
    NumPy array
        Normalised YUV image of dimensions (h,w,3).
    """

    # Rescale image to desired dimensions, if the dimensions are not already correct
    if img_dim is not None and img.shape != img_dim:
        img = reshape_image(img, img_dim)

    # Normalise the image to lie in the correct YUV ranges
    img_scaled = img / 255.

    # Convert from RGB to YUV colour space
    img_yuv = color.rgb2yuv(img_scaled)

    # Discard the U and V channels to obtain a grayscale image
    img_grayscale = img_yuv[:, :, 0]

    # Add the channel axis back to the grayscale image
    img_grayscale = np.expand_dims(img_grayscale, axis=-1)

    return img_grayscale, img_yuv


def postprocess_img(img, img_dim=None, convert_to_rgb=True):
    """Post-process an image for visualisation and general use.

    Parameters
    ----------
    img : NumPy array
        Image of dimensions (n,m,3).
    img_dim : tuple
        Dimensions (w,h) to cast image to
    convert : bool
        Flag for performing YUV to RGB conversion

    Returns
    -------
    NumPy array
        Image of dimensions (w,h,3)
    """
    
    if convert_to_rgb:
        # Convert from YUV to RGB colour space
        img = np.clip(color.yuv2rgb(img), 0.0, 1.0)

    # Scale image to convert RGB from continous to digital format
    img_scaled = np.round(img * 255).astype("uint8")

    # Resize image if a different output resolution is desired
    if img_dim is not None and img_scaled.shape != img_dim:
        img_scaled = reshape_image(img_scaled, img_dim)

    # Return image
    return img_scaled


def load_images(dir_path):
    """Load images from the directory specified by path."""
    images_in_dir = os.listdir(dir_path)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dir_path, image)), cv2.COLOR_BGR2RGB) for image in images_in_dir]

    return images


if __name__ == "__main__":
    # Show raw and preprocessed images
    image_dir = "/home/hboekema/Projects/recolour/data/tiny-imagenet-200/train/"

    images = load_images(image_dir)
    print("num images: " + str(len(images)))

    #for image in images:
    #    plt.imshow(image)
    #    plt.show()

    for image in images:
        img_pp, img_yuv_control = preprocess_img(image, img_dim=(256, 256))
        img_rgb_control = postprocess_img(img_yuv_control)

        print(np.array(image).shape)
        print(np.array(img_yuv_control).shape)
        print(np.array(img_pp).shape)

        plt.imshow(image)
        plt.show()
        plt.imshow(img_rgb_control)
        plt.show()
        plt.imshow(np.squeeze(img_pp), cmap='gray')
        plt.show()

