# Manual Implementation of Backpropagation

This is a manual implementation of backpropagation using the MNIST dataset for CSE 464 Soft Computing at New Mexico Tech.

## How to reproduce our results

### Network 1

Our first network achieved approximately 87% accuracy using a MSE cost function, sigmoid activation functions, and layers of 784 input, 100 hidden, and 10 output neurons. To run the network, make sure that Python >= 3.9.7, Numpy >= 1.20.3, Keras >= 2.9.0, and TensorFlow >= 2.9.1 are installed. Execute these commands:

`git clone https://github.com/o-oconnell/soft_computing.git`

`cd soft_computing`

`git reset --hard 903b16eec0f6f5847415d6bf9fc9856aaac0c6e8`

`python mnist.py`


On a 64-bit computer with 12 Intel Core i7 2.6 GHz cores and no GPU, the output (with `numpy.random.seed(0)` for the random weight initializations) is:

`Number of correctly classified test images after training with 60000 images:
8928
out of
10000`

The runtime using `time mnist.py` is:

`real	0m20.594s
user	1m52.485s
sys	0m1.096s` where `user` indicates that about 2 minutes of CPU time were spent executing the process without including system calls (`sys`).


### Network 2

