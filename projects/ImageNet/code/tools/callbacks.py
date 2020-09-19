
""" File with callback functions for NN training and testing  """


import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tools.images import postprocess_img


class LoggingCallback(Callback):
    """ Custom callback for visualising the network's predictions during training """
    def __init__(self, exp_dir, data, mode=True, period=1, show=False, display_dim=(256, 256)):
        """Initialisation method of the class.

        Parameters
        ----------

        """
        self.exp_dir = exp_dir
        self.period = period
        self.show = show
        self.mode = mode
        self.display_dim = display_dim

        self._format_data(data)
        self._create_vis()
        if self.mode != "val":
            self._create_log()

    def _format_data(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            self.data = {'x': data[0], 'y': data[1]}
        elif isinstance(data, dict):
            self.data = data
        else:
            raise TypeError("'data' passed to LoggingCallback is of unsupported type '{}'".format(type(data)))

    def _create_log(self):
        self.log_path = self.exp_dir + "logs/{}/loss_log.txt".format(self.mode)

        f = open(self.log_path, "w")
        f.close()

    def _create_vis(self):
        self.vis_dir = os.path.join(self.exp_dir, "vis/{}/".format(self.mode))

    def set_model(self, model):
        self.model = model

    def predict(self):
        """ Obtain (and optionally visualise) the network's predictions on the callback data """
        preds = self.model.predict(self.data['x'])
        if self.show:
            plt.imshow(preds)

        return preds

    def _store_preds(self, preds, epoch):
        """ Save the input, prediction, and GT images """
        for i in range(len(preds)):
            # Expand the Y input channels to include the U and V components (both at 0.0)
            input_y = self.data['x'][i]
            input_yuv = np.zeros(shape=(*input_y.shape[:2], 3), dtype=float)
            input_yuv[:, :, 0] = np.squeeze(input_y)

            # Calculate the RMSE difference image over the U and V channels
            pred = preds[i]
            gt = self.data['y'][i]
            rmse_channel = 2*np.sqrt(np.mean(np.square(gt[:, :, 1:] - pred[:, :, 1:]), axis=-1))
            rmse_img = np.zeros(shape=(*input_y.shape[:2], 3), dtype=float)
            rmse_img[:, :, 0] = np.clip(rmse_channel, 0.0, 1.0)

            assert(np.all(np.abs(pred[:, :, 1:]) <= 0.5))

            # Concatenate the YUV input, output, GT, and RMSE difference images
            images_to_store = [input_yuv, pred, gt, rmse_img]
            comparison = np.concatenate(images_to_store, axis=1)

            # Convert the resulting 'comparison' image to RGB
            comparison_rgb = postprocess_img(comparison, img_dim=(len(images_to_store)*self.display_dim[0], self.display_dim[1]))

            # Write the RGB 'comparison' image to file
            comparison_id = "epoch_{:05d}.comparison_{:02d}.png".format(epoch + 1, i + 1)
            cv2.imwrite(os.path.join(self.vis_dir, comparison_id), comparison_rgb)

    def on_epoch_end(self, epoch, logs=None):
        """ Store the model loss and accuracy at the end of every epoch, and store a model prediction on the callback data """
        epoch = int(epoch)

        if logs is not None and self.mode == "train":
            # Store losses in log file
            with open(self.log_path, "a") as f:
                # TODO: test and debug -- see if losses are output correctly
                f.write(json.dumps({'epoch': epoch})[:-1] + ", " + json.dumps(logs) + '\n')

        if (epoch + 1) % self.period == 0 or epoch == 0:
            # Predict on callback data
            preds = self.predict()

            # Store the predictions
            self._store_preds(preds, epoch)

