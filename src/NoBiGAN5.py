from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

from BiGAN import BiGAN
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt
import numpy as np

class NoBiGAN5(BiGAN):

    ########################################################################
    #                 Novelty detection Bidirectional GAN                  #
    ########################################################################

    def __init__(self, dim_input_g=100,
                       dim_input_d=(28, 28),
                       dim_input_l=1,
                       optimizer=Adam(beta_1=0.5),
                       random_state=36):
        """Bidirectional Generative Adversarial Network.

            Parameters
            ----------
            dim_input_g : int, default=100
                Dimension of generator input.

            dim_input_d : tuple, default=(28, 28)
                Dimension of discriminator input.

            dim_input_l : tuple, default=1
                Dimension of label input.

            optimizer : keras.optimizer, default=Adam(beta_1=0.5)
                Optimiser to use for training.
            """
        # Set input dimensions
        self.dim_input_d = dim_input_d
        self.dim_input_l = dim_input_l
        self.dim_input_g = dim_input_g

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generators
        self.generator_data  = self.build_generator_data ()
        self.generator_label = self.build_generator_label()

        # Build the encoder
        self.encoder = self.build_encoder()

        # Build and compile the combined model
        self.combined = self.build_combined()
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )

        # Define accuracy queue for printing progress
        self.accuracy_queue = [100.]*50

        # Create random state to ensure similar executions
        self.random_state = check_random_state(random_state)

        # Create class centres
        self.centres = {}


    ########################################################################
    #             Generator/Discriminator/Combined definitions             #
    ########################################################################

    def build_generator_data(self):
        """Build keras generator model for data.

            Returns
            -------
            result : keras.model
                Model to generate labels from random noise.
            """
        # Set noise and label input dimension
        noise = Input(shape=(self.dim_input_g,), name="Generator_noise")

        # Add model layers
        model = Dense(512)(noise)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dense(512)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dense(np.prod(self.dim_input_d), activation='tanh')(model)
        model = Reshape(self.dim_input_d)(model)

        # Return keras model: noise -> generator -> output
        return Model(noise, model, name="Generator_data")

    def build_generator_label(self):
        """Build keras generator model for labels.

            Returns
            -------
            result : keras.model
                Model to generate data from random noise.
            """
        # Set noise and label input dimension
        noise = Input(shape=(self.dim_input_g,), name="Generator_noise")

        # Add model layers
        model = Dense(512)(noise)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dense(512)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dense(np.prod(self.dim_input_l), activation='tanh')(model)

        # Return keras model: noise -> generator -> output
        return Model(noise, model, name="Generator_label")

    def build_discriminator(self):
        """Build keras discriminator model.

            Returns
            -------
            result : keras.model
                Model that takes discriminator input and outputs a value between
                0-1 to indicate real or generated data.
            """
        # Create data input
        data   = Input(shape=self.dim_input_d   , name="Data")
        # Create label input
        label  = Input(shape=(self.dim_input_l,), name="Label")
        # Create latent space input
        latent = Input(shape=(self.dim_input_g,), name="Latent")
        # Combine latent and data as input to discriminator
        combined = concatenate([Flatten()(data), label, latent])

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

        return Model([data, label, latent], output, name="Discriminator")

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

        # Encode unlabelled data to latent space
        e_data   = Input(shape=self.dim_input_d   , name="Data")
        e_label  = Input(shape=(self.dim_input_l,), name="Label")
        e_latent = self.encoder(e_data)

        # Build the combined model
        # The generator takes noise as input and generates fake data and labels
        g_latent = Input(shape=(self.dim_input_g,), name="Latent")
        g_data   = self.generator_data (g_latent)
        g_label  = self.generator_label(g_latent)

        # Create fake and real data
        fake = self.discriminator([g_data, g_label, g_latent])
        real = self.discriminator([e_data, e_label, e_latent])

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        return Model([e_data, e_label, g_latent], [fake, real], name="Combined")


    ########################################################################
    #                       GAN training/generating                        #
    ########################################################################

    def add_centres(self, classes):
        """Create new centres for known classes.

            Parameters
            ----------
            classes : np.array of shape=(n_classes,)
                Classes for which to create latent centres.

            Returns
            -------
            result : dict
                Dictionary of class -> latent centre coordinates.
            """
        # Initialise centres
        centres = self.centres if hasattr(self, 'centres') else {}
        # Get unknown classes
        classes = classes[~np.isin(classes, centres.keys())]
        # Create new centres
        centres_new = self.random_state.normal(0, 2*classes.shape[0],
                                              (classes.shape[0], self.dim_input_g))
        # Add new centres to existing ones
        centres.update({classes[i]: c for i, c in enumerate(centres_new)})

        # Return result
        return centres

    def train(self, X_train, y_train, classes,
                    iterations=1000,
                    batch_size=64,
                    k=1,
                    sample_interval=100,
                    verbose=True):
        """Train the BiGAN with given samples.

            Parameters
            ----------
            X_train : np.array of shape=(n_samples, dim_input_d)
                Real samples to train with.

            y_train : np.array of shape=(n_samples,)
                Real labels to train with.

            classes : np.array of shape=(n_classes,)
                classes to choose from

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
        # Add unknown classes to centres
        self.centres = self.add_centres(classes)

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

                # Select a random minibatch of images and corresponding labels
                minibatch = np.random.randint(0, X_train.shape[0], batch_size)
                X_real = X_train[minibatch]
                l_real = y_train[minibatch]
                # Generate latent dimension from input data
                z_real = self.encoder.predict(X_real)

                # Get random noise samples
                z_base = np.array([self.centres[l] for l in l_real])
                z_fake = np.random.normal(0, 1, (batch_size, self.dim_input_g))
                z_fake = z_base + z_fake
                # Generate batch of fake data and fake labels
                X_fake = self.generator_data .predict(z_fake)
                l_fake = l_real

                # Train the discriminator
                loss_real_d = self.discriminator.train_on_batch([X_real, l_real, z_real], y_real)
                loss_fake_d = self.discriminator.train_on_batch([X_fake, l_fake, z_fake], y_fake)

                # Compute the average loss
                loss_d      = 0.5* np.add(loss_real_d, loss_fake_d)

            ############################################################
            #                      Train Generator                     #
            ############################################################

            # # Get random noise samples
            # noise = np.random.normal(0, 1, (batch_size, self.dim_input_g))

            # # Train the generator to fool the discriminator, i.e. using y_real
            # loss_g = self.combined.train_on_batch(noise, y_real)

            loss_g = self.combined.train_on_batch([X_real, l_real, z_fake], [y_real, y_fake])

            # Plot progress
            if verbose:
                self.progress(loss_d[0], loss_g[0], 100*loss_d[1],
                              iteration, iterations)

            # If at save interval => save generated image samples
            if sample_interval and (iteration % sample_interval == 0):
                self.sample_images("../images/{}/{}.png".format(
                                   self.__class__.__name__, iteration))


    ########################################################################
    #                        Fit/prediction methods                        #
    ########################################################################

    def predict(self, X, y=None):
        return self.encoder(X)

    def select(self, X, y, ratio=0.8, random_state=36):
        """Randomly select classes from y to include in training.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Data corresponding to given labels.

            y : np.array of shape=(n_samples,)
                Labels corresponding to given data.

            ratio : float, default=0.8
                Ratio of labels to include in training set.

            random_state : int, RandomState instance or None, optional, default:
                36. If int, random_state is the seed used by the random number
                generator; If RandomState instance, random_state is the random

            Returns
            -------
            X : np.array of shape=(n_samples_selected, n_features)
                Selected data samples

            y : np.array of shape=(n_samples_selected)
                Selected data labels

            include : np.array of shape=(ratio*n_classes,)
                Labels included in the training data

            exclude : np.array of shape=((1-ratio)*n_classes,)
                Labels excluded from training data
            """
        # Create random state
        rs = check_random_state(random_state) if random_state else self.random_state

        # Extract all classes from labels
        classes = np.unique(y)

        # Crete the size of classes to include
        size = int(ratio * classes.shape[0])

        # Randomly select classes to include and exclude
        include = rs.choice(classes, size=size, replace=False)
        exclude = classes[~np.isin(classes, include)]

        # Get indices of data to include
        indices = np.isin(y_train, include)

        # Return result
        return X[indices], y[indices], include, exclude

    def sample_images(self, outfile, samples=5):
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
        # Get all classes
        classes = np.array([v for k, v in sorted(self.centres.items())])

        # Create subplots
        fig, axs = plt.subplots(samples, classes.shape[0])

        # Add labels to each column
        for ax, col in zip(axs[0], sorted(self.centres.keys())):
            ax.set_title(col)

        # Create as many samples per class as necessary
        for i in range(samples):
            # Generate random samples for each class
            noise = self.random_state.normal(classes, 1)
            # Generate data from noise
            data = self.generator_data.predict(noise)

            # Plot subplots
            for pos, datum in enumerate(data):
                axs[i, pos].imshow(data[pos], cmap='gray')
                axs[i, pos].axis('off')

        fig.savefig(outfile)
        plt.close()


if __name__ == '__main__':
    # Load the dataset

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Create NoBiGAN5
    gan = NoBiGAN5(dim_input_g=2, dim_input_d=(28, 28))

    # Select samples for training and novelty detection
    X_train_selected, y_train_selected, included, excluded =\
        gan.select(X_train, y_train)

    # Print which samples are selected
    print("""
    Training using {}/{} = {:5.2f}% of samples.
    Including labels: {}
    Excluding labels: {}\n\n\n\n""".format(X_train_selected.shape[0],
                                           X_train.shape[0],
                                           (100*X_train_selected.shape[0]) /
                                           X_train.shape[0],
                                           np.sort(included), np.sort(excluded)))

    # Train with selected samples
    gan.train(X_train_selected, y_train_selected, included, iterations=10000, sample_interval=100)
    # Save GAN
    gan.save('../saved/NoBiGAN5_g.h5', '../saved/NoBiGAN5_d.h5', '../saved/NoBiGAN5_c.h5')
    # Load GAN
    #gan.load('../saved/NoBiGAN5_g.h5', '../saved/NoBiGAN5_d.h5', '../saved/NoBiGAN5_c.h5')
    gan.plot_latent(X_test, y_test)
