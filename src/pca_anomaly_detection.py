from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from utils import scale, split
import numpy as np

class PCA_detector(object):

    def __init__(self, n_components=2, random_state=42):
        """PCA Anomaly Detector.

            Parameters
            ----------
            n_components : int, default=2
                Number of dimensions to reduce data to.

            random_state : int, default=42
                Random state of IsolationForest
            """
        # Initialise PCA
        self.pca = PCA(n_components)
        # Initialise anomaly detector
        self.detector = IsolationForest(contamination="auto",
                                        behaviour="new",
                                        random_state=random_state)

    def fit(self, X):
        """Fit and apply PCA dimension reduction and fit detector.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Data to fit.

            Returns
            -------
            result : self
                Returns self.
            """
        # Transform data
        X = self.pca.fit_transform(X)
        # Fit detector
        self.detector.fit(X)
        # Return self
        return self

    def predict(self, X):
        """Apply PCA dimension reduction and predict detector.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Data to predict.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Array of 0 (in case of normal) and 1 (in case of anomaly).
            """
        # Transform data
        X = self.pca.transform(X)
        # Check for outliers
        return self.detector.predict(X)

    def fit_predict(self, X):
        """Subsequently apply fit and predict methods.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Input to fit and predict.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Array of 0 (in case of normal) and 1 (in case of anomaly).
            """
        return self.fit(X).predict(X)

if __name__ == "__main__":
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Rescale -1 to 1
    X_train = scale(X_train, min=-1, max=1)
    X_test  = scale(X_test , min=-1, max=1)

    # Flatten data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test .reshape(X_test .shape[0], -1)

    # Split training data into known and unknown
    X_train_selected, y_train_selected, known, unknown = split(X_train, y_train)

    # Mark labels as known (1) and unknown (-1)
    y_train = 2*np.isin(y_train, known) - 1
    y_test  = 2*np.isin(y_test , known) - 1

    # Create PCA detection
    pca_detector = PCA_detector()

    # Train detector
    pca_detector.fit(X_train_selected)
    # Apply detector
    y_pred = pca_detector.predict(X_test)

    # Evaluate detector    
    evaluate(y_test, y_pred)
