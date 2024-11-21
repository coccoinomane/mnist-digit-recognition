import numpy as np


def get_predictions(A):
    """
    Get the predictions (Y) out of the activated output of the
    final layer (A).

    Y is a 60,000 x 1 vector with the predicted digit for
    each image.

    A is a 10 x 60,000 matrix, where each column contains the
    probabilities of each digit (0-9) for a single image.
    """
    # argmax returns the indices of the maximum values along an axis.
    return np.argmax(A, axis=0)


def get_accuracy(Y, Y_pred):
    """
    Compute the accuracy of the predictions.
    """
    return np.mean(Y == Y_pred)


def one_hot_encode(Y, n_classes):
    """
    Convert the labels of the training set into one-hot encoding,
    where n is the number of classes (10 for MNIST).
    """
    Y_one_hot = np.zeros((n_classes, Y.shape[0]))
    for i, y in enumerate(Y):
        Y_one_hot[y, i] = 1
    return Y_one_hot
