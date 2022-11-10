from keras.datasets import mnist
import numpy

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

    # update last layer of weights
    weights[len(weights)-1] -= delta_w * learning_rate

    # bias correction term
    delta_w0 = learning_rate * sk
    biases[len(biases)-1] -= delta_w0

    # move the error backwards and compute remaining weight updates
    for i in range(2, len(dimensions)):
        
        s_in = numpy.matmul(
            numpy.transpose(weights[len(weights)-i+1]),
                            s[0])
        
        si = s_in * binary_sigmoid_derivative(
            f_in_tot[len(f_in_tot)-i])
        
        s.insert(0,si)

        # Update weight (w = w - learning_rate * dC/dWjk
        # = w - alpha * s_j * x_i)
        weights[len(weights)-i+1] -=  learning_rate * numpy.matmul(
            all_activations[len(all_activations)-i+1],
                                                   numpy.transpose(si))

        # b = b - learning_rate * dC / dB
        # the rate of change with respect to the bias = error (si)
        delta_w0 = learning_rate * si
        biases[len(biases)-i] -= delta_w0

def run():

    i = 0
    while i < len(x_train):
        backpropagate(x_train[i].reshape((28*28, 1)), y_train[i], learning_rate=0.04)
        i += 1

                
    n_correct = 0
    for k in range(0, len(x_test)):
        n_correct += check_correctness(x_test[k].reshape((28*28, 1)), y_test[k])

    print("Number of correct results after " + str(i) + " values have been fed through: ")
    print(n_correct)
    print("out of")
    print(len(x_test))

        
run()
