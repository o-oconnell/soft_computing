from keras.datasets import mnist
import numpy
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

dimensions = [28*28, 100, 10]
biases = []
weights = []

numpy.random.seed(1)
for i in range(1, len(dimensions)):
    weights.append(2 * numpy.random.random_sample((dimensions[i], dimensions[i-1])) - 1)
        
for i in range(1, len(dimensions)):
    biases.append(2 * numpy.random.random_sample((dimensions[i], 1)) - 1)

def binary_sigmoid(x):
    return 1/(1.0 + numpy.exp(-x))

def binary_sigmoid_derivative(x):
    return binary_sigmoid(x) * (1 - binary_sigmoid(x))

# Takes the input and desired output,
# returns the weight and bias correction terms (which are
# the learning rate multiplied by the rate of change of the cost with respect
# to the biases/weights)

def check_correctness(inp, desired):
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
            
    if (desired == max_idx):
        return 1
    else:
        return 0
        
def backpropagate(inp, desired, learning_rate):

    if inp.shape != (dimensions[0],1):
        print("Incorrect input shape, expected 1d column array with " + str(dimensions[0]) + " entries.")

        
    # Fausett steps 2-5: feedforward and save the activations / input sums for backpropagation
    current_activation = inp
    all_activations, f_in_tot = [], []
    all_activations.append(inp)
    for i in range(0, len(weights)):
        f_in = numpy.matmul(weights[i], current_activation) + biases[i]
        f_in_tot.append(f_in)
        current_activation = binary_sigmoid(f_in)
        all_activations.append(current_activation)

    # Backpropagation of error
    # Fausett steps 6-8: backpropagation of error and updating of weights/biases

    # Array to store all errors as we move backward through the network
    s = []
    # error information term
    
    # For the MSE cost, we have
    # cost = 1/2 * (desired - actual)^2
    # actual = sigmoid(input) = sigmoid(weights * activations_of_previous_layer + biases)
    
    # dCost / dWeight = 2 * (1/2) * (desired - actual) * -sigmoid_prime(input) * activations_of_previous_layer
    # = (actual - desired) * sigmoid_prime(input) * activations_of_previous_layer

    # MSE COST:
    # ---------------------
    # sk = (all_activations[len(all_activations)-1] - desired) * binary_sigmoid_derivative(f_in_tot[len(f_in_tot)-1])
    # s.append(sk)
    # delta_w = numpy.matmul(sk,
    #                     numpy.transpose(
    #                     all_activations[len(all_activations)-2]))
    # ----------------------

    # For the cross-entropy cost, we have:
    # cost = -(desired * ln(actual) + (1 - desired) * ln(1 - actual))
    # = -(yln(a)+(1-y)ln(1-a))
    # dC/dW = -y/a*sp(wa_p+b)*a_p - (1-y)/(1-a)*-sp(wa_p+b)a_p, where a = sigmoid(w*a_p+b), and sp = derivative of sigmoid, and a_p is the activations of the previous layer
    # = sp(wa_p+b)a_p * (-y(1-a)/(a(1-a)) + a(1-y)/(a(1-a)))
    # = sp(wa_p+b)a_p * (-y(1-a) + a(1-y))/(a(1-a))
    # = sp(wa_p+b)a_p * (-y + ya + a -ya)/(a(1-a))
    # = sp(wa_p+b)a_p * (a - y)/(a(1-a))

    # Since the sigmoid function has the property that sigmoid_prime(x) = sigmoid(x)(1-sigmoid(x)), we get:
    # = a_p * (a - y)
    # (similarly to MSE, dC/db = (a-y) = sk below)

    # CROSS-ENTROPY COST
    # -----------------
    # sk = (all_activations[len(all_activations)-1] - desired) 
    # s.append(sk)
    # delta_w = numpy.matmul(sk,
    #                     numpy.transpose(
    #                     all_activations[len(all_activations)-2]))
    # -----------------

    # CROSS-ENTROPY COST WITH REGULARIZATION
    # -----------------
    # Add lambda / n_inputs * w to the update, where w is the final layer of weights
    # This means that deltaW becomes:
    # -learning_rate * (dC/dW + lambda/n_inputs * w)
    
    sk = (all_activations[len(all_activations)-1] - desired) 
    s.append(sk)
    delta_w = numpy.matmul(sk,
                        numpy.transpose(
                        all_activations[len(all_activations)-2]))
    # -----------------



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
        weights[i] -= learning_rate * (weight_changes[i] + 1.0/len(x_train) * weights[i])

    for i in range(0, len(bias_changes)):
        biases[i] -= learning_rate * bias_changes[i]

def run():
    # Reshape our images to suit a 784-neuron input layer
    x_train_0 = []
    x_test_0 = []
    for i in range(0, len(x_train)):
        x_train_0.append(x_train[i].reshape((28*28, 1)))

    for i in range(0, len(x_test)):
        x_test_0.append(x_test[i].reshape((28*28, 1)))


    # Convert our labels to one-hot encoding
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

    i = 0
    while i < len(x_train):
        backpropagate(x_train[i].reshape((28*28, 1)), y_train_0[i], learning_rate=0.03)
        i += 1

    n_correct = 0
    for k in range(0, len(x_test)):
        n_correct += check_correctness(x_test[k].reshape((28*28, 1)), y_test[k])

    print("Number of correctly classified test images after training with " + str(i) + " images:")
    print(n_correct)
    print("out of")
    print(len(x_test))

        
run()


