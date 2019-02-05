from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
from sklearn.utils import check_random_state

from BiGAN import BiGAN
from sklearn.decomposition import PCA

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
    #gan.train(X_train_selected, iterations=10000, sample_interval=None)
    # Save GAN
    #gan.save('../saved/NoBiGAN_g.h5', '../saved/NoBiGAN_d.h5', '../saved/NoBiGAN_c.h5')
    # Load GAN
    gan.load('../saved/NoBiGAN_g.h5', '../saved/NoBiGAN_d.h5', '../saved/NoBiGAN_c.h5')
    gan.plot_latent(X_test, y_test)
