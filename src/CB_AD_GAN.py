from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from BiGAN import BiGAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.utils import check_random_state
from utils import scale, split

import matplotlib.pyplot as plt
import numpy as np

class CB_AD_GAN(BiGAN):

    ########################################################################
    #                 Novelty detection Bidirectional GAN                  #
    ########################################################################

    def __init__(self, dim_input_g=100,
                       dim_input_d=(28, 28),
                       dim_input_l=1,
                       optimizer=Adam(beta_1=0.5)):
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
        model = Dense(np.prod(self.dim_input_l), activation='softmax')(model)

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

    def train(self, X_train, y_train,
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
                z_fake = np.random.normal(0, 1, (batch_size, self.dim_input_g))
                # Generate batch of fake data and fake labels
                X_fake = self.generator_data .predict(z_fake)
                l_fake = self.generator_label.predict(z_fake)

                # Train the discriminator
                loss_real_d = self.discriminator.train_on_batch([X_real, l_real, z_real], y_real)
                loss_fake_d = self.discriminator.train_on_batch([X_fake, l_fake, z_fake], y_fake)

                # Compute the average loss
                loss_d      = 0.5* np.add(loss_real_d, loss_fake_d)

            ############################################################
            #                      Train Generator                     #
            ############################################################

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

    def predict(self, X_train, X_test, random_state=42):
        """Predict whether samples in X ar anomalous based on reconstruction
            performance of the BiGAN.

            Parameters
            ----------
            X_train : np.array of shape=(n_samples, n_features)
                Samples to use for normal model.

            X_test : np.array of shape=(n_samples, n_features)
                Samples to predict.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Prediction of -1 (anomalous) or +1 (normal).
            """
        # Get latent representation of X
        z_train = self.encoder.predict(X_train)
        z_test  = self.encoder.predict(X_test )

        # Create detector
        detector = IsolationForest(contamination="auto",
                                   behaviour="new",
                                   random_state=random_state)

        # Fit training, predict testing and return result
        return detector.fit(z_train).predict(z_test)


    def generate_subsamples(self, amount=5, n_labels=8):
        """Generate output from given noise.

            Parameters
            ----------
            amount : int, default=5
                If no noise is given, generate the amount of output data given
                by this integer.

            Returns
            -------
            result : np.array of shape=(n_samples, dim_output)
                Generated data.
            """
        for i in range(10):
            # Generate some random noise
            noise = np.random.normal(0, 1, (100*amount, self.dim_input_g))

            # Generate labels
            labels = self.generator_label.predict(noise).argmax(axis=1)

            if all(np.sum(labels == l) >= amount for l in np.unique(labels)) and\
                np.unique(labels).shape[0] == n_labels:
                break

            if i == 9:
                print('Unable to generate all labels')

        # Generate data
        data = self.generator_data.predict(noise)

        # Pick amount random samples for each label
        result = {}
        for l in np.unique(labels):
            indices   = np.argwhere(labels == l).flatten()
            selected  = np.random.choice(indices, size=amount)
            result[l] = data[selected]

        # Create subplot
        fig, axs = plt.subplots(len(result), amount)

        for x, (label, data) in enumerate(sorted(result.items())):
            axs[x, 0].set_ylabel(label, rotation=0, size='large')
            for y, d in enumerate(data):
                axs[x, y].imshow(d, cmap='gray')
                axs[x, y].get_xaxis().set_visible(False)
                axs[x, y].get_yaxis().set_ticks([])

        fig.subplots_adjust(hspace=0)
        plt.show()
        exit()


if __name__ == '__main__':
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Rescale -1 to 1
    X_train = scale(X_train, min=-1, max=1)
    X_test  = scale(X_test , min=-1, max=1)

    # Select samples for training and novelty detection
    X_train_selected, y_train_selected, known, unknown = split(X_train, y_train)

    # One-hot encode values
    y_train_selected = to_categorical(y_train_selected, num_classes=10)
    #y_test           = to_categorical(y_test          , num_classes=10)
    y_test_values = y_test
    y_test  = 2*np.isin(y_test , known) - 1

    # Print which samples are selected
    print("""
    Training using {}/{} = {:5.2f}% of samples.
    Including labels: {}
    Excluding labels: {}\n\n\n\n""".format(X_train_selected.shape[0],
                                           X_train.shape[0],
                                           (100*X_train_selected.shape[0]) /
                                           X_train.shape[0],
                                           np.sort(known), np.sort(unknown)))

    # Create CB_AD_GAN
    gan = CB_AD_GAN(dim_input_g=2, dim_input_l=10, dim_input_d=(28, 28))

    # Train with selected samples - uncomment in case of retraining
    #gan.train(X_train_selected, y_train_selected, iterations=10000, sample_interval=100)
    # Save GAN
    #gan.save('../saved/CB_AD_GAN_g_test.h5', '../saved/CB_AD_GAN_d_test.h5', '../saved/CB_AD_GAN_c_test.h5')
    # Load GAN
    gan.load('../saved/CB_AD_GAN_g_50k.h5', '../saved/CB_AD_GAN_d_50k.h5', '../saved/CB_AD_GAN_c_50k.h5')

    # Predict test samples
    y_pred = gan.predict(X_train_selected, X_test)

    # Evaluate detector
    tp = np.logical_and(y_pred ==  1, y_test ==  1).sum()
    tn = np.logical_and(y_pred == -1, y_test == -1).sum()
    fp = np.logical_and(y_pred ==  1, y_test == -1).sum()
    fn = np.logical_and(y_pred == -1, y_test ==  1).sum()

    # Print result
    print("""
TP : {}
TN : {}
FP : {}
FN : {}
ACC: {}
F1 : {}""".format(tp, tn, fp, fn, (tp+tn)/(tp+tn+fp+fn), f1_score(y_test, y_pred)))

    gan.plot_latent(X_test, y_test_values, output='../images/CB_AD_GAN/latent.png')
