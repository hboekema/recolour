""" Data generator class for batched training of recolouring network """

import os
import numpy as np
import math

from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence
from tools.data import load_images


class ImageGenerator(Sequence):
    def __init__(self, img_paths, batch_size=32, img_dim=(256, 256), shuffle=True):
        # 'img_paths' is a set of filepaths to images
        self.img_paths = img_paths
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.shuffle = shuffle

        assert len(img_paths) > 0
        assert batch_size > 0
        assert np.all(np.array(img_dim) > 0)

    def __len__(self):
        # Number of batches in dataset
        return math.floor(len(self.img_paths) / self.batch_size)

    def __getitem__(self, idx):
        # Get batch at index idx
        img_paths_batch = self.img_paths[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load and return this batch of images
        img_x_batch, img_y_batch = load_images(img_paths_batch, img_dim=self.img_dim)
        return img_x_batch, img_y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_paths)


if __name__ == "__main__":
    # Test generator functionality
    image_dir = "/Users/hboekema/Desktop/recolour/data/landscapes_small/mountain/"
    images_in_dir = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image) for image in images_in_dir]

    img_gen = ImageGenerator(image_paths, batch_size=2)
    x_batch, y_batch = img_gen.__getitem__(0)
    print("x_batch shape: " + str(x_batch.shape))
    print("y_batch shape: " + str(y_batch.shape))

    for x in x_batch:
        print("x shape: " + str(x.shape))
        print("x dtype: " + str(x.dtype))
        print("x: " + str(x))
        plt.imshow(np.squeeze(x), cmap='gray')
        plt.show()

