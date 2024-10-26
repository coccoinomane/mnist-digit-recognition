from datasets import load_dataset
import numpy as np


def load_mnist_datasets(m_max: int = None, normalize: bool = False):
    """
    Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and
    return the training and test sets as numpy arrays.  Optionally limit
    the number of training samples, and normalize the pixel values to [0,1].

    The outputs are:
        - X_train: m x n matrix with the training images
        - Y_train: m x 1 vector with the training labels
        - X_test: m_test x n matrix with the test images
        - Y_test: m_test x 1 vector with the test labels
    where:
        - m is the number of training samples
        - n is the number of pixels in each image (28 x 28 = 784)
        - m_test is the number of test samples (10,000)
    """

    # Download train and test datasets
    ds = load_dataset("ylecun/mnist")
    ds_train = ds["train"]
    ds_test = ds["test"]

    # Pixels in each image
    n = 28 * 28

    # Optionally limit the number of training samples
    m = len(ds_train)

    # Convert the PIL images to flattened numpy arrays, taking into account the limit
    m = min(m, m_max) if m_max is not None else m
    X_train = np.array(ds_train[:m]["image"]).reshape(-1, n)  # m x n
    Y_train = np.array(ds_train[:m]["label"])  # m x 1

    # Do the same with the test set
    X_test = np.array(ds_test[:]["image"]).reshape(-1, n)  # m_test x n
    Y_test = np.array(ds_test[:]["label"])  # m_test x 1

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # Raise if X_train or X_test contain only zeros
    if np.all(X_train == 0) or np.all(X_test == 0):
        raise ValueError("X_train or X_test contain only zeros")

    return X_train, Y_train, X_test, Y_test


def load_minst_dataset_from_data_folder():
    """
    Load the MNIST dataset from the data folder, where it has been saved
    in CSV format.  The dataset is split into training and test sets.

    Please note that it is expected that the pixel values are normalized
    to [0,1] from their original [0,255] range, and that the first column
    is the label.  This is the format that the script save-mnist-csv.py
    saves the dataset in.
    """
    data_folder = "data/"
    try:
        X_train = np.loadtxt(
            f"{data_folder}mnist_train.csv.gz", delimiter=",", skiprows=0
        )
        X_test = np.loadtxt(
            f"{data_folder}mnist_test.csv.gz", delimiter=",", skiprows=0
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "The MNIST dataset is not available.  Please run the script save-mnist-csv.py."
        )
    Y_train = X_train[:, 0].astype(int)
    X_train = X_train[:, 1:].astype(float)
    Y_test = X_test[:, 0].astype(int)
    X_test = X_test[:, 1:].astype(float)
    return X_train, Y_train, X_test, Y_test
