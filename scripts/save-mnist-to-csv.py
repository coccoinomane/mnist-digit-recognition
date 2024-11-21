"""
Download the MNIST dataset and dump it into two CSV files in the
data folder: one for the training set and one for the test set.

Usage: python -m scripts.save-mnist-to-csv

In each CSV rows represent images; the first column is the label,
and the remaining columns are the pixel values.  The pixel values
are normalized to [0,1] from their original [0,255] range.
"""

from src.helpers.mnist import load_mnist_datasets
import numpy as np

data_folder = "data/"

# Load the MNIST dataset
X_train, Y_train, X_test, Y_test = load_mnist_datasets(normalize=True)
print(f"MNIST dataset downlaoded, X_train:", X_train.shape)

# Save the training set.  The fmt is impoertant.
np.savetxt(f"{data_folder}mnist_train.csv.gz", np.c_[Y_train, X_train], delimiter=",", fmt="%g")
print(f"Training set saved at {data_folder}mnist_train.csv.gz")

# Save the test set
np.savetxt(f"{data_folder}mnist_test.csv.gz", np.c_[Y_test, X_test], delimiter=",", fmt="%g")
print(f"Test set saved at {data_folder}mnist_test.csv.gz")
