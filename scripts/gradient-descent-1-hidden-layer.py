"""
Train a neural-network model for digit recognition on the MNIST dataset,
and get predictions out of it for a test set of 10,000 handwritten digits.

Usage: python -m scripts.gradient-descent-standalone-1-hidden-layer

The script does not make use of any deep-learning framework, as it is a
standalone implementation of a neural network with one hidden layer.

Inspired by Samson Zhang great tutorial:
- https://www.youtube.com/watch?v=w8yWXqWQYmU
With respect to Samson's tutorial, the following changes were made:
- Feat: Variable number of neurons in the hidden layer
- Feat: Use He initialization for the weights leading to faster convergence
- Fix: Bias correctly a vector now, rather than a scalar
- Fix: Sum in softmax denominator now includes each image separately, rather than
  summing over whole dataset
"""

from src.helpers.mnist import load_minst_dataset_from_data_folder
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a neural network on the MNIST dataset.")
parser.add_argument(
    "--n_hidden", type=int, default=10, help="Number of neurons in the hidden layer"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=500,
    help="Number of times the training set is passed through the network",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.5,
    help="How much the weights are updated at each iteration",
)
args = parser.parse_args()

# HYPERPARAMETERS
n_hidden = args.n_hidden
epochs = args.epochs
learning_rate = args.learning_rate

# FIXED PARAMETERS
n = 28 * 28  # Number of pixels in each image, also dimension of input layer
n_output = 10  # Number of possible outputs (0-9 digits), same as output layer
mnist_sample_size = 60_000  # Number of samples in the MNIST training set


def init_params():
    """
    Initialize weights and biases for the neural network.
    """

    # Initialization by Samson Zhang:
    # W1 = np.random.rand(n_hidden, n) - 0.5  # 32 x 784 numbers between 0 and 1
    # b1 = np.random.rand(n_hidden, 1) - 0.5  # 32 x 1 numbers between -0.5 and 0.5
    # W2 = np.random.rand(n_output, n_hidden) - 0.5  # 10 x 32 numbers between 0 and 1
    # b2 = np.random.rand(n_output, 1) - 0.5  # 10 x 1 numbers between -0.5 and 0.5

    # He initializiation suggested by Copilot:
    W1 = np.random.randn(n_hidden, n) * np.sqrt(2.0 / n)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((n_output, 1))

    # Gaussian initialization suggested by Copilot:
    # W1 = np.random.randn(n_hidden, n) * 0.01  # 32 x 784 matrix
    # b1 = np.zeros((n_hidden, 1))  # 32 x 1 vector
    # W2 = np.random.randn(n_output, n_hidden) * 0.01  # 10 x 32 matrix
    # b2 = np.zeros((n_output, 1))  # 10 x 1 vector

    return W1, b1, W2, b2


def loss(Y, A2):
    """
    Compute the error of a neural network prediction A2 with respect to
    the true values Y.

    The cross-entropy loss function is defined as:
    L = -Σᵢ yᵢ log(aᵢ)
    where yᵢ is the true value of the i-th neuron in the output layer,
    and aᵢ is the predicted value of the i-th neuron in the output layer.
    The loss is then summed over all the samples in the dataset to obtain
    a single scalar value.  More details can be found here:
    https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

    The cross-entropy is the most used loss function in classification
    problems, because it leads to a cancellation of terms in the
    backpropagation algorithm that simplifies the computation of the
    gradients.  More specifically, the cancelation happens in the
    ubiquitous term ∂L/∂z term that reduces to Y_predicted - Y_true,
    where z is the unactivated output of the final layer. More details
    at https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
    """
    return -np.sum(Y * np.log(A2))  # Add small value to avoid log(0)


def ReLU(Z):
    """
    Activation function for the input layer (and for the intermediate
    layers, if any).  Without the activation function, the layers would
    only output linear combinations of the initial input, which would make the
    whole network equivalent to a single layer.
    TODO: What if we use min(0, Z) instead of max(0, Z)?
    """
    return np.maximum(Z, 0)


def ReLU_derivative(Z):
    """
    Compute the derivative of the ReLU function.
    """
    return Z > 0


def softmax(Z):
    """
    Function that converts the output of each neuron in the final
    layer into a [0,1] probability.  Z is a 10 x 60,000 matrix, where
    each column represents the output of the final layer for a single image.
    """
    if np.any(Z > 709):
        raise ValueError("Overflow in softmax function")
    exp_Z = np.exp(Z)
    return exp_Z / (np.sum(exp_Z, axis=0, keepdims=True))


def forward_propagation(X, W1, b1, W2, b2):
    """
    Compute the forward propagation of the neural network.
    """
    # The first step will reduce the 784 pixels of each image to 32 neurons;
    # we pass the full training set of 60,000 images through the hidden layer,
    # at once, via the X matrix, transposed.  This is possible because:
    # - X is a matrix of shape 60,000 x 784
    # - W1 is a matrix of shape 32 x 784
    # - b1 is a matrix of shape 32 x 1
    # Therefore by multiplying W1 by the transposition of X, we get a matrix
    # of shape 32 x 60,000, where each column represent the output of the hidden
    # layer (and input for the hidden layer) generated by a single image.

    Z1 = W1 @ X + b1  # 32 x 60,000 unactivated output of the input layer
    A1 = ReLU(Z1)  # 32 x 60,000 activated output of the input layer
    Z2 = W2 @ A1 + b2  # 10 x 60,000 unactivated output of the hidden layer
    A2 = softmax(Z2)  # 10 x 60,000 activated output of the hidden layer
    return Z1, A1, Z2, A2


def back_propagation(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2, learning_rate=0.01):
    """
    Compute the back propagation of the neural network using thg dataset-averaged
    gradient of the loss function with respect to the weights and biases.
    """

    n_samples = X.shape[1]

    # Backpropagation (Rumelhart et al. 1986) means that we can express the
    # gradient (of the loss function with respect to the output weights W2)
    # for a single case in the dataset as the outer product of two vectors:
    #  - ∂L/∂W = δ ⊗ y
    # where δ = ∂L/∂x_last, y = ∂x_last/∂W is the activated output of the
    # penultimate layer, and x_last = W2 @ y + b2 is the unactivated output
    # of the last layer.
    # In component form, this is:
    #  - ∂L/∂W[i,j] = δ[i] * y[j].
    # Please note that δ has m elements, where m is the number of neurons
    # in the last layer (10), and y has p elements, where p is the number
    # of neurons in the previous layer (32).
    # The average of the gradient over all cases k in the dataset is then:
    #  - <∂L/∂W[i,j]> = (1/N) * Σₖ (δ[i,k] * y[j,k])
    # where N is the number of cases in the dataset.
    # By realizing that the product of two matrices in position i,j is just
    # the scalar product of the i-th row of the first matrix and the j-th
    # column of the second matrix, we can write the average gradient as:
    #  - <∂L/∂W> = (1/N) * δM @ yM.T
    # where δM and yM are matrices where each row is the δ and y vector
    # for a single case in the dataset, respectively.
    # Please note that δM has shape m x N (10 x 60000) and yM has shape
    # N x p (32 x 60,000), hence the resulting gradient has shape m x p
    # (10 x 32) as expected.
    # The main achievement of backpropagation is that we can compute the
    # gradient of the loss function with respect to the weights and biases
    # of the network for all cases in the dataset at once, by using matrix
    # multiplication.
    δM = A2 - Y
    # ^^^ δM = ∂L/∂x_last has the very simple form A2 - Y thanks to the choice
    # of the cross-entropy loss function and the softmax activation function,
    # see the comment above the loss function definition.
    dL_dW2 = δM @ A1.T / n_samples

    # Check that the gradient has the expected shape
    if dL_dW2.shape != W2.shape:
        raise ValueError(f"Gradient dL_dW2 shape {dL_dW2.shape} does not match W2 shape {W2.shape}")

    # The gradient wrt to the bias for a single datapoint is just the δ vector
    # itself, because ∂L/∂b_i = ∂L/∂x_last * ∂x_last/∂b_i = δ_i * 1 = δ_i.
    # The average gradient over all cases in the dataset is then:
    #  - <∂L/∂b_i> = (1/N) * Σₖ δ[i,k]
    # This is just the sum over the rows of δM, resulting in a m x 1 vector.
    dL_db2 = np.sum(δM, axis=1, keepdims=True) / n_samples  # keepdims to ensure column vector

    # Check that the gradient has the expected shape
    if dL_db2.shape != b2.shape:
        raise ValueError(f"Gradient dL_db2 shape {dL_db2.shape} does not match b2 shape {b2.shape}")

    # The gradient wrt to the weights of the first layer is given by:
    δM_penultimate = W2.T @ δM * ReLU_derivative(Z1)
    dL_dW1 = δM_penultimate @ X.T / n_samples

    # Check that the gradient has the expected shape
    if dL_dW1.shape != W1.shape:
        raise ValueError(f"Gradient dL_dW1 shape {dL_dW1.shape} does not match W1 shape {W1.shape}")

    # The gradient wrt to the bias for the first layer is given by:
    dL_db1 = np.sum(δM_penultimate, axis=1, keepdims=True) / n_samples

    # Check that the gradient has the expected shape
    if dL_db1.shape != b1.shape:
        raise ValueError(f"Gradient dL_db1 shape {dL_db1.shape} does not match b1 shape {b1.shape}")

    # Update the parameters
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    # Debug: print the magnitude of the gradients compared to the weights
    # print("Gradient magnitudes:")
    # print(" - dL_dW2:", np.linalg.norm(dL_dW2) / np.linalg.norm(W2))
    # print(" - dL_db2:", np.linalg.norm(dL_db2) / np.linalg.norm(b2))
    # print(" - dL_dW1:", np.linalg.norm(dL_dW1) / np.linalg.norm(W1))
    # print(" - dL_db1:", np.linalg.norm(dL_db1) / np.linalg.norm(b1))

    return W1, b1, W2, b2


def get_predictions(A2):
    """
    Get the predictions (Y) out of the activated output of the
    final layer (A2).

    Y is a 60,000 x 1 vector with the predicted digit for
    each image.

    A2 is a 10 x 60,000 matrix, where each column contains the
    probabilities of each digit (0-9) for a single image.
    """
    # argmax returns the indices of the maximum values along an axis.
    return np.argmax(A2, axis=0)


def get_accuracy(Y, Y_pred):
    """
    Compute the accuracy of the predictions.
    """
    return np.mean(Y == Y_pred)


def loss_function(Y, A2):
    """
    Compute the loss function of the neural network.
    This is how wrong the network is in its predictions.
    For a given image, it is given by the difference between
    the predicted probability A2, a vector with 10 elements,
    and the actual probability Y, a vector with 9 zero elements
    and a single 1 element in the position of the actual digit.
    """
    return A2 - Y  # 10 x 60,000 matrix


def one_hot_encode(Y):
    """
    Convert the labels of the training set into one-hot encoding.
    """
    Y_one_hot = np.zeros((n_output, Y.shape[0]))
    for i, y in enumerate(Y):
        Y_one_hot[y, i] = 1
    return Y_one_hot


def print_params_info(W1, b1, W2, b2):
    print(f" - W1: {W1.shape}, min: {np.min(W1)}, max: {np.max(W1)}, mean: {np.mean(W1)}")
    print(f" - b1: {b1.shape}, min: {np.min(b1)}, max: {np.max(b1)}, mean: {np.mean(b1)}")
    print(f" - W2: {W2.shape}, min: {np.min(W2)}, max: {np.max(W2)}, mean: {np.mean(W2)}")
    print(f" - b2: {b2.shape}, min: {np.min(b2)}, max: {np.max(b2)}, mean: {np.mean(b2)}")


def train(X, Y, W1, b1, W2, b2, epochs=1000):
    """
    Train the neural network.
    """
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = back_propagation(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2, learning_rate)
        if epoch % 25 == 0:
            print(f"Epoch {epoch}")
            print(f" - Loss: {np.mean(loss(Y, A2))}")
            Y_pred = get_predictions(A2)
            acc = get_accuracy(Y_train, Y_pred)
            print(f" - Accuracy: {acc}")
            print_params_info(W1, b1, W2, b2)
    return W1, b1, W2, b2


# Load the MNIST dataset
X_train, Y_train, X_test, Y_test = load_minst_dataset_from_data_folder()

# We shall work with matrices where each row represents a single sample image
X_train = X_train.T
X_test = X_test.T
if X_train.shape != (n, mnist_sample_size):
    raise ValueError(
        f"X_train shape {X_train.shape} does not match expected shape {n, mnist_sample_size}"
    )

print(f"MNIST dataset loaded:")
print(f" - X_train: {X_train.shape}, min: {np.min(X_train)}, max: {np.max(X_train)}")
print(f" - Y_train: {Y_train.shape}, min: {np.min(Y_train)}, max: {np.max(Y_train)}")
print(f" - X_test: {X_test.shape}, min: {np.min(X_test)}, max: {np.max(X_test)}")
print(f" - Y_test: {Y_test.shape}, min: {np.min(Y_test)}, max: {np.max(Y_test)}")
print(f" - one hot encoding of Y_train: {one_hot_encode(Y_train).shape}")

# Get initial parameters
W1, b1, W2, b2 = init_params()
print("Parameters initialized:")
print_params_info(W1, b1, W2, b2)

# Feed the samples to the neural network for the first time
Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
print("Fist forward propagation done:")
print(f" - Z1: {Z1.shape}, min: {np.min(Z1)}, max: {np.max(Z1)}, mean: {np.mean(Z1)}")
print(f" - A1: {A1.shape}, min: {np.min(A1)}, max: {np.max(A1)}, mean: {np.mean(A1)}")
print(f" - Z2: {Z2.shape}, min: {np.min(Z2)}, max: {np.max(Z2)}, mean: {np.mean(Z2)}")
print(f" - A2: {A2.shape}, min: {np.min(A2)}, max: {np.max(A2)}, mean: {np.mean(A2)}")

# Get first prediction and visually check the accuracy is 10%
Y_pred = get_predictions(A2)
acc = get_accuracy(Y_train, Y_pred)
print("Accuracy without training should be around 10%:", acc)

# Train the neural network
Y_train_one_hot = one_hot_encode(Y_train)
W1, b1, W2, b2 = train(X_train, Y_train_one_hot, W1, b1, W2, b2, epochs=epochs)

# Feed the training samples to the neural network
Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
print("Final accuracy on TRAINING set:")
Y_pred = get_predictions(A2)
acc = get_accuracy(Y_train, Y_pred)
print(f" - Accuracy: {acc}")

# Feed the test samples to the neural network
Z1, A1, Z2, A2 = forward_propagation(X_test, W1, b1, W2, b2)
print("Final accuracy on TEST set:")
Y_pred = get_predictions(A2)
acc = get_accuracy(Y_test, Y_pred)
print(f" - Accuracy: {acc}")
