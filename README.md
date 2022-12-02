# Manual Implementation of Backpropagation

This is a manual implementation of backpropagation using the MNIST dataset for CSE 464 Soft Computing at New Mexico Tech.

## How to reproduce our results

To run each of our networks, make sure that Python >= 3.9.7, Numpy >= 1.20.3, Keras >= 2.9.0, and TensorFlow >= 2.9.1 are installed. Clone and enter our repo:

`git clone https://github.com/o-oconnell/soft_computing.git`

`cd soft_computing`

### Network 1

Our first network achieved approximately 89% accuracy using a MSE cost function, sigmoid activation functions, and layers of 784 input, 100 hidden, and 10 output neurons. In our repository, execute these commands:

`git reset --hard d0d028a3c9319fcfa2be7ecd71d66ae6e293ab4b`

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

Our second network achieves near 94% accuracy using a cross-entropy cost function, sigmoid activation functions, and layers of 784 input, 100 hidden, and 10 output neurons. In our repository, execute these commands:

`git reset --hard 2923f616f2d0119a3315eece5b7e58f92875d1d5`

`python mnist.py`

On a 64-bit computer with 12 Intel Core i7 2.6 GHz cores and no GPU, the output of `time mnist.py` (with `numpy.random.seed(0)` for the random weight initializations) is:

`Number of correctly classified test images after training with 60000 images:
9358
out of
10000`

`real	0m19.259s
user	1m44.624s
sys	0m1.048s`


