
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


def cross_entropy_from_logits(*args, **kwargs):
    """ Syntactic sugar for BinaryCrossentropy function from logits """
    return BinaryCrossentropy(from_logits=True)(*args, **kwargs)


def discriminator_loss(real_output, fake_output):
    """ Disciminator cross-entropy loss (from logits) """
    real_loss = cross_entropy_from_logits(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    """ Generator cross-entropy loss (from logits) """
    return cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)


def wasserstein_discriminator_loss(real_output, fake_output):
    """ Discriminator Wasserstein loss (from logits) """
    return tf.reduce_mean(fake_output - real_output)


def wasserstein_generator_loss(fake_output):
    """ Generator Wasserstein loss (from logits) """
    return -tf.reduce_mean(fake_output)

