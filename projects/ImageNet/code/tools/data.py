
""" File for handling data loading and saving. """


import os
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tools.images import preprocess_img


def gather_images_in_dir(img_dir):
    """ Collect the paths of all the images ina given directory """
    image_ids = os.listdir(img_dir)
    image_paths = [os.path.join(img_dir, image_id) for image_id in image_ids]

    return image_paths


def load_images(img_paths, img_dim=(256, 256)):
    """Load RGB images for which the paths are given. Convert to YUV form and preprocess them for model training/prediction.

    Parameters
    ----------
    img_paths : list of str
        List of valid paths to images to load
    img_dim : tuple of int
        Image dimensions to load images into

    Returns
    -------
    list
        Lists of x (input) and of y (GT) images
    """
    img_x_batch = []
    img_y_batch = []
    for img_path in img_paths:
        # Load image not PIL format and convert to NumPy array
        img = img_to_array(load_img(img_path, target_size=img_dim))

        # Pre-process the image to obtain the (x,y) data pairs
        img_x, img_y = preprocess_img(img)

        img_x_batch.append(img_x)
        img_y_batch.append(img_y)

    return np.array(img_x_batch), np.array(img_y_batch)

