# Manual Implementation of Backpropagation

This is a manual implementation of backpropagation using the MNIST dataset for CSE 464 Soft Computing at New Mexico Tech.

## How to reproduce our results

### Network 1

Our first network achieved approximately 87% accuracy using a MSE cost function, sigmoid activation functions, and layers of 784, 30 (hidden), and 10 neurons. To run the network, make sure that Python >= 3.9.7, Numpy >= 1.20.3, Keras >= 2.9.0, and TensorFlow >= 2.9.1 are installed. Execute these commands:

`git clone https://github.com/o-oconnell/soft_computing.git`

`cd soft_computing`

`git reset --HARD 903b16eec0f6f5847415d6bf9fc9856aaac0c6e8`

`python mnist.py`

### Network 2

