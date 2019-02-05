from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
from sklearn.utils import check_random_state

from BiGAN import BiGAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

class NoBiGAN(BiGAN):

    ########################################################################
    #                 Novelty detection Bidirectional GAN                  #
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
        rs = check_random_state(random_state)

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


    # Create NoBiGAN
    gan = NoBiGAN(dim_input_g=2, dim_input_d=(28, 28))

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
    gan.train(X_train_selected, iterations=10000, sample_interval=None)
    # Save GAN
    #gan.save('../saved/NoBiGAN_g.h5', '../saved/NoBiGAN_d.h5', '../saved/NoBiGAN_c.h5')
    # Load GAN
    gan.load('../saved/NoBiGAN_g.h5', '../saved/NoBiGAN_d.h5', '../saved/NoBiGAN_c.h5')
    gan.plot_latent(X_test, y_test)
