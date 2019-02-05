from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

from GAN import GAN

import matplotlib.pyplot as plt
import numpy as np

class BiGAN(GAN):

    ########################################################################
    #             Bidirectional Generative Adversarial Network             #
    ########################################################################

    def __init__(self, dim_input_g=100,
                       dim_input_d=(28, 28),
                       optimizer=Adam(beta_1=0.5)):
        """Bidirectional Generative Adversarial Network.

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

        # Build the encoder
        self.encoder = self.build_encoder()

        # Build the rest of the network
        super(BiGAN, self).__init__(dim_input_g, dim_input_d, optimizer)


    ########################################################################
    #             Generator/Discriminator/Combined definitions             #
    ########################################################################

    def build_encoder(self):
        """Build keras encoder model.

            Returns
            -------
            result : keras.model
                Model to encode data to latent space from original input.
            """
        # Initialise model
        model = Sequential()

        # Add model layers
        model.add(Flatten(input_shape=self.dim_input_d))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.dim_input_g))

        # Set input shape
        input = Input(shape=self.dim_input_d)

        # Return keras model: input -> encoder -> output
        return Model(input, model(input), name="Encoder")

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
        model.add(Dense(512, input_dim=self.dim_input_g))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.dim_input_d), activation='tanh'))
        model.add(Reshape(self.dim_input_d))

        # Set noise input dimension
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
        # Create latent space input
        latent = Input(shape=(self.dim_input_g,))
        # Create data input
        data   = Input(shape=self.dim_input_d)
        # Combine latent and data as input to discriminator
        combined = concatenate([latent, Flatten()(data)])

        # Create discriminator model
        model = Dense(1024)(combined)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        output = Dense(1, activation="sigmoid")(model)

        return Model([latent, data], output, name="Discriminator")

    def build_combined(self):
        """Build model by combining Generator and Discriminator.

            Returns
            -------
            result : keras.model
                Model that goes from input -> generator -> discriminator.
                Where the discriminator part will not be trained.
            """
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Build the combined model
        # The generator takes noise as input and generates fake data
        g_latent = Input(shape=(self.dim_input_g,))
        g_data   = self.generator(g_latent)

        # Encode data to latent space
        e_data   = Input(shape=self.dim_input_d)
        e_latent = self.encoder(e_data)

        # Create fake and real data
        fake = self.discriminator([g_latent, g_data])
        real = self.discriminator([e_latent, e_data])

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        return Model([g_latent, e_data], [fake, real], name="Combined")


    ########################################################################
    #                       GAN training/generating                        #
    ########################################################################

    def train(self, X_train, iterations=1000,
                             batch_size=64,
                             k=1,
                             sample_interval=100,
                             verbose=True):
        """Train the BiGAN with given samples.

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
                z_fake = np.random.normal(0, 1, (batch_size, self.dim_input_g))
                # Generate batch of fake data
                X_fake = self.generator.predict(z_fake)

                # Select a random minibatch of images
                minibatch = np.random.randint(0, X_train.shape[0], batch_size)
                X_real = X_train[minibatch]
                z_real = self.encoder.predict(X_real)

                # Train the discriminator
                loss_real_d = self.discriminator.train_on_batch([z_real, X_real], y_real)
                loss_fake_d = self.discriminator.train_on_batch([z_fake, X_fake], y_fake)
                # Compute the average loss
                loss_d      = 0.5* np.add(loss_real_d, loss_fake_d)

            ############################################################
            #                      Train Generator                     #
            ############################################################

            # # Get random noise samples
            # noise = np.random.normal(0, 1, (batch_size, self.dim_input_g))

            # # Train the generator to fool the discriminator, i.e. using y_real
            # loss_g = self.combined.train_on_batch(noise, y_real)

            loss_g = self.combined.train_on_batch([z_fake, X_real], [y_real, y_fake])

            # Plot progress
            if verbose:
                self.progress(loss_d[0], loss_g[0], 100*loss_d[1],
                              iteration, iterations)

            # If at save interval => save generated image samples
            if sample_interval and (iteration % sample_interval == 0):
                self.sample_images("../images/{}/{}.png".format(
                                   self.__class__.__name__, iteration))

    ########################################################################
    #                        Visualisation methods                         #
    ########################################################################

    def plot_latent(self, X, y=None, output=None):
        """Plot X when mapped to latent space.

            Parameters
            ----------
            X : torch.Tensor of shape(n_samples, dim_input)
               Input variables to propagate through the network.

            y : torch.Tensor of shape(n_samples,), optional
               Labels of x, if given show the labels of x.

            output : string, optional
               If given write image to output file.
            """
        # Apply encoding layer
        X = self.encoder.predict(X)
        # Convert to numpy array
        y = np.zeros(X.shape[0]) if y is None else y

        # Raise warning if latent space has too many dimensions.
        if X.shape[1] != 2:
            warnings.warn("Latent space has dimension {}. "
                          "Reducing dimension to 2 using PCA.".format(
                          X.shape[1]), RuntimeWarning)

            # Reduce to 2 dimensions
            X = PCA(n_components=2).fit_transform(X)

        # Plot each label as a specific colour
        for y_ in np.unique(y):
            # Get samples from x with given label
            X_ = X[y == y_]
            # Plot samples from x
            plt.scatter(X_[:, 0], X_[:, 1], label=y_)

        # Show plot
        plt.legend()
        if output is None:
            plt.show()
        else:
            plt.savefig(output)


if __name__ == '__main__':
    # Load the dataset

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    gan = BiGAN(dim_input_g=2, dim_input_d=(28, 28))
    #gan.train(X_train, iterations=10000, sample_interval=100)
    #gan.save('../saved/BiGAN_g.h5', '../saved/BiGAN_d.h5', '../saved/BiGAN_c.h5')
    gan.load('../saved/BiGAN_g.h5', '../saved/BiGAN_d.h5', '../saved/BiGAN_c.h5')
    gan.plot_latent(X_test, y_test)
