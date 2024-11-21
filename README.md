Pedagogical script that trains a neural-network model for digit recognition, using a single hidden layer with a variable number of neurons.

The training dataset is [the MNIST dataset](http://yann.lecun.com/exdb/mnist/) of 60,000 handwritten digits; the script gets predictions and estimates the accuracy using the separate MNIST test set of 10,000 handwritten digits.

The script improves on the [excellent tutorial by Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU) and as such does not make use of any deep-learning framework.

All datasets are included in the data folder.

# Usage

```bash
python -m scripts.one-hidden-layer --n_hidden 10 --epochs 500 --learning_rate 0.2 --initial_params he
```

where:

- `n_hidden` is the number of neurons in the hidden layer
- `epochs` is the number of times the training set is passed through the network
- `learning_rate` is how much the weights are updated at each iteration


# Accuracy

To obtain a 92% accuracy with the same amount of time, run:

```bash
python -m scripts.one-hidden-layer --n_hidden 10 --learning_rate 0.2 --epochs 500 --initial_params he
```

Increase the number of hidden layers or epochs to get more accuracy, although it seems that convergence slows down a lot around 95%.


# Inspiration

To reproduce the results of [Samson's tutorial](https://www.youtube.com/watch?v=w8yWXqWQYmU), run:

```bash
python -m scripts.one-hidden-layer --n_hidden 10 --learning_rate 0.1 --epochs 500 --initial_params samson
```

To run the exact script from the tutorial, use the following command:

```bash
python -m scripts.samson
```

Both commands should give an accuracy of around 85% on the test set.

With respect to Samson's tutorial, the following changes were made:

- Feat: Variable number of neurons in the hidden layer
- Feat: Use He initialization for the weights leading to faster convergence
- Fix: Bias correctly a vector now, rather than a scalar
- Fix: Sum in softmax denominator now includes each image separately, rather than
  summing over whole dataset

