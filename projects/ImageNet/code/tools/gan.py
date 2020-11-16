""" GAN training procedure model """


import numpy as np
import tensorflow as tf
import tqdm
import collections

from queue import Queue
from tools.replay import ReplayBuffer


class GANModel:
    def __init__(self,
            generator,
            discriminator,
            generator_loss,
            discriminator_loss,
            gen_optimizer,
            disc_optimizer,
            gen_runs_per_epoch,
            disc_runs_per_epoch,
            apply_gp=False,
            gp_weight=10,
            replay_buffer_epochs=None,
            ):

        self.generator = generator
        self.generator_loss = generator_loss
        self.gen_optimizer = gen_optimizer
        self.gen_runs_per_epoch = gen_runs_per_epoch
        
        self.discriminator = discriminator
        self.discriminator_loss = discriminator_loss
        self.disc_optimizer = disc_optimizer
        self.disc_runs_per_epoch = disc_runs_per_epoch

        self.apply_gp = apply_gp
        self.gp_weight = gp_weight

        self.replay_buffer_epochs = replay_buffer_epochs

    def __call__(self, inputs):
        """ Return tensor of generated images """
        return self.generator(inputs)

    def predict(self, inputs):
        """ Alternative to __call__ that returns numpy array instead of tensor """
        return self.generator.predict(inputs)

    def _gradient_penalty(self, generated_images, real_images):
        """ Apply gradient penalty to regularise the discriminator, making it
        meet the Lipschitz condition (required for WGANs) """
        batch_size = generated_images.shape[0]
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)

        interpolated_images = tf.constant(alpha * generated_images + (1 - alpha) * real_images)
        
        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(interpolated_images)
            interpolated_score = self.discriminator(interpolated_images, training=True)

        interpolated_gradients = gradient_tape.gradient(interpolated_score, [interpolated_images])[0]
        slopes = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(interpolated_gradients), axis=[1, 2, 3]))
        
        gradient_penalty = tf.math.reduce_mean(tf.math.square(slopes - 1.))
        return gradient_penalty

    def _initialise_replay_buffer(self, max_episodes, episode_length):
        self.replay_buffer = ReplayBuffer(batches_to_replay, episode_length)

    def _use_replay(self, new_entries): 
        # Use replay buffer to stabilise training
        self.replay_buffer.update(generated_images)
        return self.replay_buffer.draw()

    def _discriminator_train_step(self, data):
        x, y = data

        with tf.GradientTape() as disc_tape:
            # Generate images, score these and a set of real images, and calculate the discriminator loss
            generated_images = self.generator(x, training=True)
            if self.replay_buffer_epochs is not None:
                generated_images = self._use_replay(generated_images)
            real_output = self.discriminator(y, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
            if self.apply_gp:
                gp_loss = self._gradient_penalty(generated_images, y)
                disc_loss = disc_loss + self.gp_weight * gp_loss

        # Get gradients and apply them
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return disc_loss

    def _generator_train_step(self, data):
        x, y = data

        with tf.GradientTape() as gen_tape:
            # Generate images, score these, and calculate the generator loss
            generated_images = self.generator(x, training=True)
            fake_output = self.discriminator(generated_images, training=False)
            gen_loss = self.generator_loss(fake_output)

        # Get gradients and apply them
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return gen_loss

    def _set_train_mode(self, x):
        if isinstance(x, tf.keras.utils.Sequence):
            # Generator was passed to the function
            self.train_mode = 'data_generator'
        else:
            # Data was passed to the function
            self.train_mode = 'data'

    def _get_data_batch(self, x, y, batch_size, index):
        if self.train_mode == "data_generator":
            return x.__getitem__(index)
        elif self.train_mode == "data":
            # TODO: implement
            assert False, "not implemented" 
        else:
            raise ValueError("Data input not recognised.")

    def _shuffle_data(self, x, y):
        if self.train_mode == "data_generator":
            x.shuffle_data()
            return x, None
        elif self.train_mode == "data":
            # TODO: implement
            assert False, "not implemented" 
        else:
            raise ValueError("Data input not recognised.") 

    def _run_callbacks(self, callbacks, epoch, logs, on):
        for callback in callbacks:
            callback.set_model(self.generator)
            callback.set_critic(self.discriminator)
            if on == "epoch_end":
                callback.on_epoch_end(epoch, logs)
            elif on == "epoch_begin":
                callback.on_epoch_begin(epoch, logs)


    def _perform_training_epoch(self, x, y, epoch, batch_size, steps_per_epoch):
        """ Perform single training epoch (consisting of discriminator training and generator training runs) """
        print("\nTraining discriminator...")
        for disc_run_number in range(self.disc_runs_per_epoch):
            print("Run {}/{}".format(disc_run_number + 1, self.disc_runs_per_epoch))
            with tqdm.tqdm(total=steps_per_epoch) as disc_pbar:
                for step in range(steps_per_epoch):
                    # Get batch of data for this train step
                    data = self._get_data_batch(x, y, batch_size, step)
                    
                    # Train discriminator
                    disc_loss = self._discriminator_train_step(data)
                    disc_loss = disc_loss.numpy()

                    # Update progress bar
                    disc_pbar.set_postfix({"disc_loss": np.round(float(disc_loss), decimals=3)}, refresh=True)
                    disc_pbar.update()

            tqdm.tqdm.close(disc_pbar)
            
            # Shuffle data in generator
            x, y = self._shuffle_data(x, y)
            
    
        print("\nTraining generator...")
        for gen_run_number in range(self.gen_runs_per_epoch):
            print("Run {}/{}".format(gen_run_number + 1, self.gen_runs_per_epoch))
            with tqdm.tqdm(total=steps_per_epoch) as gen_pbar:
                for step in range(steps_per_epoch):
                    # Get batch of data for this train step
                    data = self._get_data_batch(x, y, batch_size, step)
                    
                    # Train generator
                    gen_loss = self._generator_train_step(data)
                    gen_loss = gen_loss.numpy()

                    # Update progress bar
                    gen_pbar.set_postfix({"gen_loss": np.round(float(gen_loss), decimals=3)}, refresh=True)
                    gen_pbar.update()

            tqdm.tqdm.close(gen_pbar)
            
            # Shuffle data in generator
            x, y = self._shuffle_data(x, y)

        return gen_loss, disc_loss

    def train(self, x=None, y=None, batch_size=None, epochs=1, callbacks=None, validation_data=None, steps_per_epoch=None):
        """ Custom training loop for adversarial training """
        self._set_train_mode(x)
        losses = collections.OrderedDict()
        loss_history = {}

        if self.replay_buffer_epochs is not None:
            # Initialise replay buffer
            print("Training discriminator with replay")
            batches_to_replay = steps_per_epoch * self.replay_buffer_epochs
            self._initialise_replay_buffer(batches_to_replay, batch_size)

        # Start training loop
        for epoch in range(1, epochs+1):
            print("\nEpoch %d" % epoch)
            # Call the model callbacks
            self._run_callbacks(callbacks, epoch, losses, "epoch_begin")
            
            # Train the networks
            gen_loss, disc_loss = self._perform_training_epoch(x, y, epoch, batch_size, steps_per_epoch)

            # Store most recent losses 
            losses["gen_loss"] = "{:.03f}".format(gen_loss)
            losses["disc_loss"] = "{:.03f}".format(disc_loss)
            
            # Print losses for this epoch
            print("Epoch losses:\tgen_loss: {:.03f}\tdisc_loss: {:.03f}".format(gen_loss, disc_loss))

            # Perform validation run
            # TODO: write validation pass

            # Call the model callbacks
            self._run_callbacks(callbacks, epoch, losses, "epoch_end")

            # Update the loss history
            loss_history[str(epoch)] = losses
        
        return loss_history


    def fit(self, *args, **kwargs):
        self.train(*args, **kwargs)

