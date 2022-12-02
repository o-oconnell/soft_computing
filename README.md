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

On a 64-bit computer with 12 Intel Core i7-10750H  2.6 GHz cores and no GPU, the output (with `numpy.random.seed(0)` for the random weight initializations) is:

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

On a 64-bit computer with 12 Intel Core i7-10750H 2.6 GHz cores and no GPU, the output of `time mnist.py` (with `numpy.random.seed(0)` for the random weight initializations) is:

`Number of correctly classified test images after training with 60000 images:
9358
out of
10000`

`real	0m19.259s
user	1m44.624s
sys	0m1.048s`


### Modified Network 2

modified_bp.py is a modified version of network 2. This version only performs the BP step on an input if the network guesses incorrectly on said input.

The repository can be brought to the state it was in during this testing using the command `git reset --hard 181520e6c247aa040576e6ce5b0d1517234bbc15`
The program can then be run using `python modified_bp.py`

On a 64-bit computer with a Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz and no GPU, the output of `modified_bp.py` (with `numpy.random.seed(0)` for the random weight initializations) is:
Completed round 0
Training took 5.745041370391846 seconds
Number of correctly classified test images after training with 60000 images:
9153
out of
10000

This is compared to `mnist.py` (network 2), which has the following output:
Completed round 0
Training took 16.20714521408081 seconds
Number of correctly classified test images after training with 60000 images:
9358
out of
10000

### Number of Rounds Control

By modifying the code, it is possible to change how many times the training set it fed through the network. This is done by changing the numerical value on line 189 of modified_bp.py. The default number of rounds is 1.


### Network 3

Using the cross-entropy cost function with L2 regularization we achieved a signiificant improvement over network 2. You can get the correct version of `mnist.py` using `git reset --hard 044747e7fd10f5fc39d9813beaf91848b86171de`.

With a seeded random number generator we recorded results:

`time python mnist.py`

`Number of correctly classified test images after training with 60000 images:
9422
out of
10000`

`real	0m23.874s
user	2m12.152s
sys	0m1.104s`
