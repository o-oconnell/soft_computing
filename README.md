# Manual Implementation of Backpropagation

This is a manual implementation of backpropagation using the MNIST dataset for CSE 464 Soft Computing at New Mexico Tech. It is a three layer neural network with sigmoid activation functions on all neurons after the input layer.

This implementation achieves an accuracy of around 87%.



The current implementation has a list of fixed/to-fix issues:

1. We updated the weights continuously instead of waiting until all of the errors have been calculated. This is a problem because we need the original weights in order to backpropagate the error.
For a hidden layer, the error S_in_j = sum over k of Sk * Wjk, so if we've changed Wjk prior to calculating S_in_j, we're not backpropagating the error according to the proof.

2. Our y_train and y_test vectors were encoded as single integers. We needed an array of 10 values with a single one in order to execute the algorithm properly.

3. Kind of an issue (still have not fixed): we are initializing the weights to small random values from a uniform distribution. It would be preferable to initialize them from a standard normal distribution as values would be clustered around zero and less likely to saturate.

