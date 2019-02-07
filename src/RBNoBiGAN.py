from keras.datasets import mnist
from sklearn.metrics import f1_score

from BiGAN import BiGAN
from utils import scale, split
import numpy as np

class RBNoBiGAN(BiGAN):

    ########################################################################
    #                 Novelty detection Bidirectional GAN                  #
    ########################################################################

    def predict(self, X, threshold=0.8):
        """Predict whether samples in X ar anomalous based on reconstruction
            performance of the BiGAN.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Samples to predict.

            threshold : float, default=0.8
                Maximum MSE to be concidered normal.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Prediction of -1 (anomalous) or +1 (normal).
            """
        # Rescale X to range -1 to 1
        X = scale(X, min=-1, max=1)
        # Get latent representation of X
        z = self.encoder.predict(X)
        # Reconstruct output of X
        r = self.generator_data.predict(z)

        # Compute MSE between original and reconstructed
        mse = np.square(X - r).reshape(X.shape[0], -1).mean(axis=1)

        # Apply threshold for prediction
        predict = 2*(mse <= threshold) - 1

        # Return result
        return predict


if __name__ == '__main__':
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Rescale -1 to 1
    X_train = scale(X_train, min=-1, max=1)
    X_test  = scale(X_test , min=-1, max=1)

    # Split training data into known and unknown
    _, _, known, unknown = split(X_train, y_train)
    X_train_known = X_train[np.isin(y_train, known)]

    # Mark labels as known (1) and unknown (-1)
    y_train = 2*np.isin(y_train, known) - 1
    y_test  = 2*np.isin(y_test , known) - 1

    # Create RBNoBiGAN
    gan = RBNoBiGAN(dim_input_g=2, dim_input_d=(28, 28))
    # Train with selected samples
    #gan.train(X_train_selected, iterations=10000, sample_interval=None)
    # Save GAN
    #gan.save('../saved/RBNoBiGAN_g.h5', '../saved/RBNoBiGAN_d.h5', '../saved/RBNoBiGAN_c.h5')
    # Load GAN
    gan.load('../saved/RBNoBiGAN_g.h5', '../saved/RBNoBiGAN_d.h5', '../saved/RBNoBiGAN_c.h5')

    # Predict test samples
    y_pred = gan.predict(X_test, threshold=0.9)

    # Evaluate detector
    tp = np.logical_and(y_pred ==  1, y_test ==  1).sum()
    tn = np.logical_and(y_pred == -1, y_test == -1).sum()
    fp = np.logical_and(y_pred ==  1, y_test == -1).sum()
    fn = np.logical_and(y_pred == -1, y_test ==  1).sum()

    # Print result
    print("""
TP:  {}
TN:  {}
FP:  {}
FN:  {}
ACC: {}
F1 : {}""".format(tp, tn, fp, fn, (tp+tn)/(tp+tn+fp+fn), f1_score(y_test, y_pred)))
