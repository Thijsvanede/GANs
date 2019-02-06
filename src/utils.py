from sklearn.utils import check_random_state
import numpy as np

def split(X, y, ratio=0.8, random_state=36):
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
    indices = np.isin(y, include)

    # Return result
    return X[indices], y[indices], include, exclude
