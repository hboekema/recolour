""" GAN training procedure model """


import numpy as np
import tensorflow as tf
import tqdm
import collections


class GANModel:
    def __init__(self, generator, discriminator, generator_loss, discriminator_loss, gen_optimizer, disc_optimizer,
            gen_runs_per_epoch, disc_runs_per_epoch):
        self.generator = generator
        self.generator_loss = generator_loss
        self.gen_optimizer = gen_optimizer
        self.gen_runs_per_epoch = gen_runs_per_epoch
        
        self.discriminator = discriminator
        self.discriminator_loss = discriminator_loss
        self.disc_optimizer = disc_optimizer
        self.disc_runs_per_epoch = disc_runs_per_epoch

    def __call__(self, inputs):
        return self.generator.predict(inputs)

    def predict(self, inputs):
        """ Alternative to __call__ """
        return self(inputs)

    def _discriminator_train_step(self, data):
        x, y = data

        with tf.GradientTape() as disc_tape:
            # Generate images, score these and a set of real images, and calculate the discriminator loss
            generated_images = self.generator(x, training=False)
            real_output = self.discriminator(y, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)

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

