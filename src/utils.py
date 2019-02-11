from sklearn.metrics import f1_score
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

def scale(X, min=0, max=1):
    """Scale X to given range.

        Parameters
        ----------
        X : np.array of shape=(n_samples, n_features)
            Data to scale.

        min : float, default = 0
            Minimum value to scale to.

        max : float, default = 1
            Maximum value to scale to.

        Returns
        -------
        result : np.array of shape=(n_samples, n_features)
            Scaled data.
        """
    # Scale data to min - max
    return (X - X.min()) / (X.max() - X.min()) * (max - min) + min

def evaluate(y_true, y_pred):
    """Prints evaluation report.

        Parameters
        ----------
        y_true : np.array of shape=(n_samples,)
            Actual values of data, -1 for unknown or 1 for known.

        y_pred : np.array of shape=(n_samples,)
            Predicted values of data, -1 for unknown or 1 for known.
        """
    # Compute True/False Positives/Negatives
    tp = np.logical_and(y_pred ==  1, y_true ==  1).sum()
    tn = np.logical_and(y_pred == -1, y_true == -1).sum()
    fp = np.logical_and(y_pred ==  1, y_true == -1).sum()
    fn = np.logical_and(y_pred == -1, y_true ==  1).sum()

    # Print result
    print()
    print("Evaluation report")
    print("--------------------------")
    print("  True  Positives: {:>6}".format(tp))
    print("  True  Negatives: {:>6}".format(tn))
    print("  False Positives: {:>6}".format(fp))
    print("  False Negatives: {:>6}".format(fn))
    print("  Accuracy       : {:>.4f}".format((tp+tn)/(tp+tn+fp+fn)))
    print("  F1-score       : {:>.4f}".format(f1_score(y_true, y_pred)))
    print()
