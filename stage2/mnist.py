from keras.datasets import mnist
import numpy
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

dimensions = [28*28, 30, 10]
biases = []
weights = []

# Step 0

for i in range(1, len(dimensions)):
    weights.append(2 * numpy.random.random_sample((dimensions[i], dimensions[i-1])) - 1)
        
for i in range(1, len(dimensions)):
    biases.append(2 * numpy.random.random_sample((dimensions[i], 1)) - 1)


# See p.324 of Fausett's artificial neural systems, we are using a quadratic cost.
def quadratic_E(target, found):
    return 0.5 * (target - found)**2

def E_prime(target, found):
    return (target - found)

def binary_sigmoid(x):
    return 1/(1.0 + numpy.exp(-x))

def binary_sigmoid_derivative(x):
    return binary_sigmoid(x) * (1 - binary_sigmoid(x))

# Takes the input and desired output,
# returns the weight and bias correction terms (which are
# the learning rate multiplied by the rate of change of the cost with respect
# to the biases/weights)

def check_correctness(inp, desired):
    # print("input shape is "+str(inp.shape))
    
    current_activation = inp
    all_activations, f_in_tot = [], []
    all_activations.append(inp)
    for i in range(0, len(weights)):
        f_in = numpy.matmul(weights[i], current_activation) + biases[i]
        f_in_tot.append(f_in)
        current_activation = binary_sigmoid(f_in)

    max_idx = -1
    max_v = -2
    for i in range(0, len(current_activation)):
        if current_activation[i] > max_v:
            max_idx = i
            max_v = current_activation[i]
            
    # print("desired result " + str(desired))
    # print("found result " + str(max_idx))
    # print("current activation:")
    # print(current_activation)
    if (desired == max_idx):
        # print("max idx " + str(max_idx))
        # print("desired " + str(desired))
        # print("activations: " + str(current_activation))
        return 1
    else:
        # print("max idx " + str(max_idx))
        # print("desired " + str(desired))
        # print("activations:"+str(current_activation))
        return 0
        
def backpropagate(inp, desired, learning_rate):

    if inp.shape != (dimensions[0],1):
        print("Incorrect input shape, expected 1d column array with " + str(dimensions[0]) + " entries.")

        
    # Step 2-5: feedforward and save the activations / input sums for backpropagation
    current_activation = inp
    all_activations, f_in_tot = [], []
    all_activations.append(inp)
    for i in range(0, len(weights)):
        f_in = numpy.matmul(weights[i], current_activation) + biases[i]
        f_in_tot.append(f_in)
        current_activation = binary_sigmoid(f_in)
        all_activations.append(current_activation)

    """
    wantedVal = numpy.argmax(desired)
    max_idx = numpy.argmax(current_activation)

    if(wantedVal == max_idx):
        return None
        """
        
        
    # Backpropagation of error
    # Step 6-8: backpropagation of error and updating of weights/biases

    # Array to store all errors as we move backward through the network
    s = []
    # error information term
    sk = (all_activations[len(all_activations)-1] - desired) * binary_sigmoid_derivative(f_in_tot[len(f_in_tot)-1])
    s.append(sk)
    
    # weight correction term
    delta_w = numpy.matmul(sk,
                        numpy.transpose(
                        all_activations[len(all_activations)-2]))

    weight_changes = []
    bias_changes = []
    
    # update last layer of weights
    weight_changes.insert(0, delta_w)

    # bias correction term
    delta_w0 = sk
    bias_changes.insert(0, delta_w0)

    # move the error backwards and compute remaining weight updates
    for i in range(2, len(dimensions)):
        
        s_in = numpy.matmul(
            numpy.transpose(weights[len(weights)-i+1]),
                            s[0])
        
        si = s_in * binary_sigmoid_derivative(
            f_in_tot[len(f_in_tot)-i])

        s.insert(0,si) # to compute errors of prior layer correctly

        # Update weight (w = w - learning_rate * dC/dWjk
        # = w - alpha * s_j * x_i)
        weight_changes.insert(0, numpy.matmul(si, numpy.transpose(all_activations[len(all_activations)-i-1])))

        # b = b - learning_rate * dC / dB
        # the rate of change with respect to the bias = error (si)
        delta_w0 = si
        bias_changes.insert(0, delta_w0)

    for i in range(0, len(weight_changes)):
        weights[i] -= learning_rate * weight_changes[i]

    for i in range(0, len(bias_changes)):
        biases[i] -= learning_rate * bias_changes[i]

def run():

    # Issues with our algorithm:
    # 1. We update the weights as we go instead of waiting until all of the errors have been calculated. This is a problem because we need the original weights in order to backpropagate the error.
    # For a hidden layer, the error S_in_j = sum over k of Sk * Wjk, so if we've changed Wjk prior to calculating S_in_j, we're not backpropagating the error according to the proof.

    # 2. Our y_train and y_test vectors were encoded as single integers. We needed an array of 10 values with a single one in order to execute the algorithm properly.

    # 3. Kind of an issue (still have not fixed): we are initializing the weights to small random values from a uniform distribution. It would be preferable to initialize them from a standard normal distribution as values would be clustered around zero and less likely to saturate.

    x_train_0 = []
    x_test_0 = []
    for i in range(0, len(x_train)):
        x_train_0.append(x_train[i].reshape((28*28, 1)))

    for i in range(0, len(x_test)):
        x_test_0.append(x_test[i].reshape((28*28, 1)))

    y_train_0 = []
    y_test_0 = []
    for i in range(0, len(y_train)):
        add = numpy.zeros((10, 1))
        add[y_train[i]] = 1.0
        y_train_0.append(add)
    for i in range(0, len(y_test)):
        add = numpy.zeros((10, 1))
        add[y_test[i]] = 1.0
        y_test_0.append(add)
    
    startTime = time.time()
    for rounds in range(100):
        i = 0
        while i < len(x_train):
            backpropagate(x_train[i].reshape((28*28, 1)), y_train_0[i], learning_rate=0.03)
            i += 1
        print("Completed round " + str(rounds))
    print("Test took " + str(time.time() - startTime) + " seconds")
    

    n_correct = 0
    for k in range(0, len(x_test)):
        n_correct += check_correctness(x_test[k].reshape((28*28, 1)), y_test[k])

    print("Number of correct results after " + str(i) + " values have been fed through: ")
    print(n_correct)
    print("out of")
    print(len(x_test))

        
run()


