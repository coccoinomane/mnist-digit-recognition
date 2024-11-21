import numpy as np


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
