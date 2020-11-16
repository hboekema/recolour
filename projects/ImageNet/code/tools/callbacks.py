
""" File with callback functions for NN training and testing  """


import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

from tools.images import postprocess_img, write_to_img
from tools.metrics import sigmoid_np


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

        self.model = None
        self.critic = None

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

    def set_critic(self, critic):
        self.critic = critic

    def predict(self):
        """ Obtain (and optionally visualise) the network's predictions on the callback data """
        assert self.model is not None

        preds = self.model.predict(self.data['x'])
        if self.show:
            plt.imshow(preds)

        return preds

    def _score_preds(self, preds):
        """ Score the given predictions using a pre-selected critic model """
        assert self.critic is not None

        pred_scores = self.critic.predict(preds)
        # TODO: generalise this code neatly - should be an option to specify logits vs probits
        #pred_scores = sigmoid_np(pred_scores) 

        return pred_scores

    def _add_UV_channels_to_image(self, image_index):
        # Expand the Y input channels to include the U and V components (both at 0.0)
        input_y = self.data['x'][image_index]
        input_yuv = np.zeros(shape=(*input_y.shape[:2], 3), dtype=float)
        input_yuv[:, :, 0] = np.squeeze(input_y)

        return input_y, input_yuv

    def _compute_RMS_error_image(self, preds, image_index):
        # Calculate the RMSE difference image over the U and V channels
        pred = preds[image_index]
        assert(np.all(np.abs(pred[:, :, 1:]) <= 0.5))
            
        gt = self.data['y'][image_index]
        rmse_channel = 2*np.sqrt(np.mean(np.square(gt[:, :, 1:] - pred[:, :, 1:]), axis=-1))
        rmse_img = np.zeros(shape=(*gt.shape[:2], 3), dtype=float)
        rmse_img[:, :, 0] = np.clip(rmse_channel, 0.0, 1.0)

        return rmse_img, pred, gt

    def _store_preds(self, preds, epoch, scores={}):
        """ Save the input, prediction, and GT images """
        images_to_write = []
        for image_index in range(len(preds)):
            # Add UV channels back to Y-channel input image
            _, input_yuv = self._add_UV_channels_to_image(image_index)
            
            # Compute difference image for visual error comparison
            rmse_img, pred, gt = self._compute_RMS_error_image(preds, image_index)

            # Postprocess the YUV input, output, GT, and RMSE difference images
            input_rgb = postprocess_img(input_yuv, img_dim=self.display_dim, convert_to_rgb=True)
            pred_rgb = postprocess_img(pred, img_dim=self.display_dim, convert_to_rgb=True) 
            gt_reshaped = postprocess_img(gt, img_dim=self.display_dim, convert_to_rgb=True) 
            rmse_img_reshaped = postprocess_img(rmse_img, img_dim=self.display_dim, convert_to_rgb=True) 
            
            # Label the output and GT images with their score, if provided
            if scores:
                pred_score = scores["pred_scores"][image_index]
                pred_score_text = "Score: {:.03f}".format(*pred_score)
                pred_rgb = write_to_img(pred_rgb, pred_score_text)
                
                gt_score = scores["gt_scores"][image_index]
                gt_score_text = "Score: {:.03f}".format(*gt_score)
                gt_reshaped = write_to_img(gt_reshaped, gt_score_text)

            # Concatenate into a 'comparison' image
            images_to_store = [input_rgb, pred_rgb, gt_reshaped, rmse_img_reshaped]
            comparison_img = np.concatenate(images_to_store, axis=1)
            images_to_write.append(comparison_img)

        # Write the RGB 'comparison' images to file as a single, composite image
        composite_img = np.concatenate(images_to_write, axis=0)
        composite_id = "epoch_{:05d}.comparison.png".format(epoch + 1, image_index + 1)
        cv2.imwrite(os.path.join(self.vis_dir, composite_id), cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))

    def _predict_and_store(self, epoch):
        # Predict on callback data
        preds = self.predict()

        # Critique (score) predictions and GT data if critic model was set
        scores = {}
        if self.critic is not None:
            scores["pred_scores"] = self._score_preds(preds)
            scores["gt_scores"] = self._score_preds(self.data['y'])            

        # Store the predictions
        self._store_preds(preds, epoch, scores)

    def _write_logs_to_file(self, epoch, logs):
        # Store losses in log file
        with open(self.log_path, "a") as f:
            # TODO: test and debug -- see if losses are output correctly
            f.write(json.dumps({'epoch': epoch})[:-1] + ", " + json.dumps(logs) + '\n')

    def on_epoch_begin(self, epoch, logs=None):
        epoch = int(epoch) 
        if epoch == 1:
            self._predict_and_store(epoch=0)

    def on_epoch_end(self, epoch, logs=None):
        """ Store the model loss and accuracy at the end of every epoch, and store a model prediction on the callback data """
        epoch = int(epoch)

        if logs is not None and self.mode == "train":
            self._write_logs_to_file(epoch, logs)

        if epoch % self.period == 0:
            self._predict_and_store(epoch)
            
