from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

class GAN(object):

    ########################################################################
    #                    Generative Adversarial Network                    #
    ########################################################################

    def __init__(self, dim_input_g=100,
                       dim_input_d=(28, 28),
                       optimizer=Adam(beta_1=0.5)):
        """Generative Adversarial Network.

            Parameters
            ----------
            dim_input_g : int, default=100
                Dimension of generator input.

            dim_input_d : tuple, default=(28, 28)
                Dimension of discriminator input.

            optimizer : keras.optimizer, default=Adam(beta_1=0.5)
                Optimiser to use for training.
            """
        # Set input dimensions
        self.dim_input_d = dim_input_d
        self.dim_input_g = dim_input_g

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # Build and compile the combined model
        self.combined = self.build_combined()
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )

        # Define accuracy queue for printing progress
        self.accuracy_queue = [100.]*50


    ########################################################################
    #             Generator/Discriminator/Combined definitions             #
    ########################################################################

    def build_generator(self):
        """Build keras generator model.

            Returns
            -------
            result : keras.model
                Model to generate data from random noise.
            """
        # Initialise model
        model = Sequential()

        # Add model layers
        model.add(Dense(256, input_dim=self.dim_input_g))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.dim_input_d), activation='tanh'))
        model.add(Reshape(self.dim_input_d))

        # Set noise input dimension.
        noise = Input(shape=(self.dim_input_g,))

        # Return keras model: noise -> generator -> output
        return Model(noise, model(noise), name="Generator")

    def build_discriminator(self):
        """Build keras discriminator model.

            Returns
            -------
            result : keras.model
                Model that takes discriminator input and outputs a value between
                0-1 to indicate real or generated data.
            """
        # Initialise model
        model = Sequential()

        # Add model layers
        model.add(Flatten(input_shape=self.dim_input_d))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Set input dimension
        input   = Input(shape=self.dim_input_d)

        # Return keras model: input -> discriminator -> output
        return Model(input, model(input), name="Discriminator")

    def build_combined(self):
        """Build model by combining Generator and Discriminator.

            Returns
            -------
            result : keras.model
                Model that goes from input -> generator -> discriminator.
                Where the discriminator part will not be trained.
            """
        # Build the combined model
        # The generator takes noise as input and generates fake data
        input_g   = Input(shape=(self.dim_input_g,))
        generated = self.generator(input_g)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated data as input and discriminates
        output_d = self.discriminator(generated)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        return Model(input_g, output_d, name="Combined")


    ########################################################################
    #                       GAN training/generating                        #
    ########################################################################

    def train(self, X_train, iterations=1000,
                             batch_size=64,
                             k=1,
                             sample_interval=100,
                             verbose=True):
        """Train the Generative Adversarial Network with given samples.

            Parameters
            ----------
            X_train : np.array of shape=(n_samples, dim_input_d)
                Real samples to train with.

            iterations : int, default=1000
                Number of iterations to use for training.

            batch_size : int, default=64
                Number of samples in each batch, 1 batch is used per iteration.

            k : int, default=1
                Number of discriminator updates per generator update.

            sample_interval : int, default=100
                Iteration interval at which to output randomly generated
                results of generator.

            verbose : boolean, default=True
                If verbose is set, print current status.
            """
        # Generated image samples
        if sample_interval:
            self.sample_images("../images/{}/{}.png".format(
                               self.__class__.__name__, 0))

        # Rescale -1 to 1
        X_train = X_train / (X_train.max() / 2.) - 1.

        # Adversarial ground truths
        y_real = np.ones ((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        # Loop over all iterations
        for iteration in range(1, iterations+1):

            ############################################################
            #                    Train Discriminator                   #
            ############################################################

            # Apply multiple discriminator updates per generator update
            for _ in range(k):

                # Get random noise samples
                noise = np.random.normal(0, 1, (batch_size, self.dim_input_g))
                # Generate batch of fake data
                X_fake = self.generator.predict(noise)

                # Select a random minibatch of images
                minibatch = np.random.randint(0, X_train.shape[0], batch_size)
                X_real = X_train[minibatch]

                # Train the discriminator
                loss_real_d = self.discriminator.train_on_batch(X_real, y_real)
                loss_fake_d = self.discriminator.train_on_batch(X_fake, y_fake)
                # Compute the average loss
                loss_d      = 0.5* np.add(loss_real_d, loss_fake_d)

            ############################################################
            #                      Train Generator                     #
            ############################################################

            # Get random noise samples
            noise = np.random.normal(0, 1, (batch_size, self.dim_input_g))

            # Train the generator to fool the discriminator, i.e. using y_real
            loss_g = self.combined.train_on_batch(noise, y_real)

            # Plot progress
            if verbose:
                self.progress(loss_d[0], loss_g, 100*loss_d[1],
                              iteration, iterations)

            # If at save interval => save generated image samples
            if sample_interval and (iteration % sample_interval == 0):
                self.sample_images("../images/{}/{}.png".format(
                                   self.__class__.__name__, iteration))

    def generate(self, noise=None, amount=5):
        """Generate output from given noise.

            Parameters
            ----------
            noise : np.array of shape=(n_samples, dim_input_generator), optional
                If given, generate output from given noise.

            amount : int, default=5
                If no noise is given, generate the amount of output data given
                by this integer.

            Returns
            -------
            result : np.array of shape=(n_samples, dim_output)
                Generated data.
            """
        # Create noise if required
        if noise is None:
            noise = np.random.normal(0, 1, (amount, self.dim_input_g))
        # Return prediction
        return self.generator.predict(noise)


    ########################################################################
    #                              I/O methods                             #
    ########################################################################

    def save(self, out_gen, out_dis, out_com):
        """Saves weights of GAN to outfile.

            Parameters
            ----------
            out_gen : string
                Path to output file for generator.

            out_dis : string
                Path to output file for discriminator.

            out_com : string
                Path to output file for combined model.
            """
        # Save weights to outfiles
        self.generator    .save_weights(out_gen)
        self.discriminator.save_weights(out_dis)
        self.combined     .save_weights(out_com)

    def load(self, in_gen, in_dis, in_com):
        """Loads weights of GAN from infile.

            Parameters
            ----------
            in_gen : string
                Path to input file for generator.

            in_dis : string
                Path to input file for discriminator.

            in_com : string
                Path to input file for combined model.
            """
        # Load weights from infiles
        self.generator    .load_weights(in_gen)
        self.discriminator.load_weights(in_dis)
        self.combined     .load_weights(in_com)


    ########################################################################
    #                  Visualisation/information methods                   #
    ########################################################################

    def sample_images(self, outfile, data=None, width=5, height=5):
        """Generate width x height images and write them to outfile.

            Parameters
            ----------
            outfile : string
                Path to outfile to write image to.

            width : int, default=5
                Number of generated images in width of output figure.

            height : int, default=5
                Number of generated images in height of output figure.
            """
        # Generate random images
        if data is None:
            X_fake = self.generate(amount=(height * width))
        else:
            X_fake = data

        # Rescale images 0 - 1
        X_fake = (X_fake - X_fake.min()) / (X_fake.max() - X_fake.min())

        # Create subplot
        fig, axs = plt.subplots(height, width)
        counter = 0
        for x in range(height):
            for y in range(width):
                axs[x, y].imshow(X_fake[counter], cmap='gray')
                axs[x, y].axis('off')
                counter += 1
        fig.savefig(outfile)
        plt.close()

    def progress(self, d_loss, g_loss, acc, iteration, total):
        """Method for printing current progress

            Parameters
            ----------
            d_loss : float
                Loss of Discriminator

            g_loss : float
                Loss of Generator

            acc : float
                Accuracy of Discriminator

            iteration : int
                Current iteration

            total : int
                Total number of iterations used.
            """
        # Compute average accuracy of last N samples
        self.accuracy_queue = self.accuracy_queue[1:] + [acc]
        acc = sum(self.accuracy_queue) / len(self.accuracy_queue)

        # Clear 6 lines
        print("\x1b[2K\033[F\x1b[2K\033[F\x1b[2K\033[F"
              "\x1b[2K\033[F\x1b[2K\033[F\x1b[2K\033[F", end='')

        # Compute variables to use in format string
        progress = round(40*iteration/total)
        p_str = '#'*progress
        r_str = '.'*(40-progress)
        progress = 100*iteration/total
        length = len(str(total))

        # Print progress
        print(f"""

Discriminator accuracy: {acc:>6.2f}%
Discriminator     loss: {d_loss:.5f}
Generator         loss: {g_loss:.5f}
Progress: [{p_str}{r_str}] {iteration:{length}}/{total} = {progress:>6.2f}%"""
             )


if __name__ == '__main__':
    # Load the dataset

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    gan = GAN(dim_input_g=2, dim_input_d=(28, 28))
    gan.train(X_train, iterations=10000, sample_interval=100)
    gan.save('../saved/GAN_g.h5', '../saved/GAN_d.h5', '../saved/GAN_c.h5')
