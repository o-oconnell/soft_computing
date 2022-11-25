from keras.datasets import mnist
import numpy
import time
import random

#The class defining a neuron (not input neurons)
class Neuron:
    input_weights = None
    bias = None
    last_output = None
    last_inputs = None
    last_delta = None

    #Makes a new neuron, with the provided number of input weights
    def __init__(this, num_inputs):
        this.input_weights = numpy.random.random_sample(num_inputs) * (0.5/(28 * 28))
        this.bias = numpy.random.random()

    #Gives the output of the neuron for an input list
    def processInput(this, input):

        #Auto-return None if the size of the input != the size of the weight list
        if(len(input) != len(this.input_weights)):
            print("Neuron got input that did not match it's weight size")
            print("Expected: " + str(len(this.input_weights)) + "\nReceived: " + str(len(input)))
            return None

        #Sum the weight*input
        weighted_in = numpy.multiply(input, this.input_weights)
        totalSum = sum(weighted_in)

        #Add the bias, then squash
        totalSum += this.bias
        totalSum = binary_sigmoid(totalSum)

        #Save the result, then return it
        this.last_output = totalSum
        this.last_inputs = input
        return totalSum

    #Corrects the weights according to the error noted
    def correctWeights(this, learning_rate, target = None, affectedDeltas = None):
        #Check to make sure either target or affectedDeltas were given
        if(target == None and affectedDeltas == None):
            print("Either target or affected deltas must be given to weight correction!")
            return None

        #First, calculate the delta
        delta = 0
        #If this is an output neuron, delta is calculated from the target and the output
        if(affectedDeltas == None):
            delta = -(target - this.last_output) * this.last_output * (1 - this.last_output)

        #If this is a hidden neuron, delta is based on the affected deltas and the out
        else:
            delta = sum(affectedDeltas) * this.last_output * (1 - this.last_output)

        this.last_delta = delta

        #Next, calculate the error for each weight, and correct them
        #Calculate the error
        errors = numpy.multiply(this.last_inputs, delta)

        #Correct each weight
        weight_changes = numpy.multiply(errors, learning_rate)
        this.input_weights = numpy.subtract(this.input_weights, weight_changes)

        #Correct the bias
        this.bias -= delta * learning_rate

def binary_sigmoid(x):
    return 1/(1.0 + numpy.exp(-x))

#This initializes all of the layers of the network, according to the provided dimensions
def initialize(dimensions):
    #Make a list to store all of the layers
    neurons = []

    #Loop through each of the provided dimensions
    for dimnum in range(1, len(dimensions)):
        layer = []

        #Make neurons for the layer, making sure they have the right # of inputs (should be dimension of previous layer)
        for i in range(dimensions[dimnum]):
            layer.append(Neuron(dimensions[dimnum - 1]))

        #Add the layer to the larger neuron list
        neurons.append(layer)

    #Return the neuron matrix
    return neurons

#This feeds the given input through the network
def feed_forward(input, neurons):
    #Make lists to hold the inputs & outputs of each layer
    currentInput = []
    currentOutput = input #Input is considered the output of the input layer

    #Loop through each layer of neurons
    for layer in neurons:
        #Make the old output the new input, clear the old output
        currentInput = currentOutput
        currentOutput = []
        
        #Feed the input to each neuron in the layer, record the outputs
        for neuron in layer:
            currentOutput.append(neuron.processInput(currentInput))

    #Return the overall result
    return currentOutput

#This backpropagates the system for the given input
def backprop(input, label, learning_rate, neurons):

    #Feed the input forward through the network
    feed_forward(input, neurons)

    #Loop through each layer, starting from the output
    for layerNum in range(len(neurons) - 1, -1, -1):
        layer = neurons[layerNum]

        #Do one thing if this is the output layer
        if(layerNum == len(neurons) - 1):
            #Loop through all of the neurons in the layer
            for i in range(len(layer)):
                #Correct the error depending on if this neuron should output 1 or -1
                if i == label:
                    layer[i].correctWeights(learning_rate, target = 1)
                else:
                    layer[i].correctWeights(learning_rate, target = 0)

        #And another thing if it isn't
        else:
            #Gather the deltas of all neurons in the previous layer
            prevDeltas = []
            for neuron in neurons[layerNum - 1]:
                prevDeltas.append(neuron.last_delta)

            #Correct the errors of all neurons in this layer, using the gathered deltas
            for neuron in layer:
                neuron.correctWeights(learning_rate, affectedDeltas = prevDeltas)

#Counts how many of the given input the network classifies correctly
def countCorrect(input, labels, neurons):
    #Loop through each input
    count = 0
    for inputIndex in range(len(input)):
        #Feed the current input through the network
        values = feed_forward(input[inputIndex], neurons)

        #Figure out which neuron lit up the brightest
        maxIndex = -1
        maxValue = -1
        for i in range(len(values)):
            if(values[i] > maxValue):
                maxValue = values[i]
                maxIndex = i

        #Check if it was correct
        if(maxIndex == labels[inputIndex]):
            count += 1

    #Print the result
    print("The model got " + str(count) + "/" + str(len(input)) + " correct")

#Load & normalize the data
(raw_x_train, y_train), (raw_x_test, y_test) = mnist.load_data()
(raw_x_train, raw_x_test) = (raw_x_train / 255.0, raw_x_test / 255.0)

#Take only part of the training set, so we can debug faster
raw_x_train = raw_x_train[:100]
y_train = y_train[:100]

#Reshape the images
x_train = [None] * len(raw_x_train)
x_test = [None] * len(raw_x_test)
for imageIndex in range(len(raw_x_train)):
    x_train[imageIndex] = raw_x_train[imageIndex].reshape(28 * 28)
for imageIndex in range(len(raw_x_test)):
    x_test[imageIndex] = raw_x_test[imageIndex].reshape(28 * 28)

#Set the dimensions of the different layers
dimensions = [28*28, 30, 10]

#Initialize the neuron matrix
neurons = initialize(dimensions)

print("Ready to start training!")

#Do a whole bunch of backprop, iterating a set # of times
iterations = 1000
for i in range(iterations):
    startTime = time.time()

    #Shuffle the training set
    temp = list(zip(x_train, y_train))
    random.shuffle(temp)
    x_train, y_train = zip(*temp)
    x_train, y_train = list(x_train), list(y_train)

    for inputIndex in range(len(x_train)):
        backprop(x_train[inputIndex], y_train[inputIndex], 0.5, neurons)
    print("Completed round " + str(i))
    print("It took " + str(time.time() - startTime) + " seconds")

print("Training set")
countCorrect(x_train, y_train, neurons)
print("Test set")
countCorrect(x_test, y_test, neurons)