from keras.datasets import mnist
from sklearn.metrics import f1_score

from BiGAN import BiGAN
from utils import split
import numpy as np

class NoBiGAN(BiGAN):

    ########################################################################
    #                 Novelty detection Bidirectional GAN                  #
    ########################################################################

    def predict(self, X, threshold=0.8):
        """Predict """
        # Rescale X to range -1 to 1
        X = 2 * ((X - X.min()) / (X.max() - X.min()) - 0.5)
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
    X_train = X_train / (X_train.max() / 2.) - 1.
    X_test  = X_test  / (X_test .max() / 2.) - 1.

    # Split training data into known and unknown
    _, _, known, unknown = split(X_train, y_train)
    X_train_known = X_train[np.isin(y_train, known)]

    # Mark labels as known (1) and unknown (-1)
    y_train = 2*np.isin(y_train, known) - 1
    y_test  = 2*np.isin(y_test , known) - 1

    # Create NoBiGAN
    gan = NoBiGAN(dim_input_g=2, dim_input_d=(28, 28))
    # Train with selected samples
    #gan.train(X_train_selected, iterations=10000, sample_interval=None)
    # Save GAN
    #gan.save('../saved/NoBiGAN_g.h5', '../saved/NoBiGAN_d.h5', '../saved/NoBiGAN_c.h5')
    # Load GAN
    gan.load('../saved/NoBiGAN_g.h5', '../saved/NoBiGAN_d.h5', '../saved/NoBiGAN_c.h5')

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
