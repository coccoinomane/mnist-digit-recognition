import numpy as np


def cross_entropy_loss(Y, A):
    """
    Compute the error of a neural network prediction A with respect to
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
    return -np.sum(Y * np.log(A))  # Add small value to avoid log(0)
