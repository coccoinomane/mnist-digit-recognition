"""
Show an image from the MNIST dataset, and print its label.

Usage: show-mnist-image.py <index> <split>

where
  -  <index> is the index of the image to show (0-59999)
  -  <split> is either 'train' or 'test' to select the specific MNIST dataset
  
Example:
    $ python -m scripts.show-mnist-image 5 train

Requires the 'datasets' library:
    $ pip install datasets

Github Gist available at:
  - https://gist.github.com/coccoinomane/8de713a06b0b9cb43b621197f60ecb75
"""

import sys
from datasets import load_dataset

usage = "Usage: python -m scripts.show-mnist-image <index> <split>"

if len(sys.argv) != 3:
    print(usage)
    sys.exit(1)

# Parse index argument
index = int(sys.argv[1])
if index < 0 or index >= 60000:
    print("Index out of range, MNIST dataset has 60000 images.")
    print(usage)
    sys.exit(1)

# Parse split
split = sys.argv[2]
if split not in ["train", "test"]:
    print("Split must be either 'train' or 'test'.")
    print(usage)
    sys.exit(1)

# Load the dataset
ds = load_dataset("ylecun/mnist", split=split)

# Print the image label
print("Label:", ds[index]["label"])

# Get the image, in PIL format
image = ds[index]["image"]

# Show the image
image.show()
