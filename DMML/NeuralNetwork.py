import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(y):
    return y * (1-y)

# A general NN with specifiable amout of layers and perceptrons per layer
class NeuralNetwork:
    # layers is an array of numbers, indicating the number of nodes for each layer
    # where layers[0] is the input and layers[-1] the output layer
    def __init__(self, *layers):
        self.weights = []
        self.bias = []

        prevLayer = layers[0]
        
        # for all but the first layer: create weights and biases
        for layerIx in range(1, len(layers)):
            nOfLayer = layers[layerIx]
            self.weights.append(np.random.uniform(-1, 1, (nOfLayer, prevLayer)))
            self.bias.append(np.random.uniform(-1, 1, (nOfLayer)).reshape(-1, 1))
            prevLayer = layers[layerIx]

        # print("W=",self.weights, "B=",self.bias)

        self.learningRate = 0.1

        # set activation function & it's d/dx
        self.activationFunc = sigmoid
        self.activationFuncD = dsigmoid

    def predict(self, startLayer):
        # the layer n-1, starting with the input-layer
        prevLayer = np.array(startLayer).reshape(-1, 1)

        # for each layer n, calculate it's output for the next layer
        for curernt_layer in range(len(self.weights)):
            layerOut = np.dot(self.weights[curernt_layer], prevLayer)
            layerOut += self.bias[curernt_layer]
            layerOut = np.vectorize(self.activationFunc)(layerOut)

            prevLayer = layerOut

        # return the last layer (aka the output-layer)'s output
        return prevLayer


    def train(self, startLayer, target_arr):
        # save all layer's output, starting with the input layer (which just outputs the input)
        prevLayer = [np.array(startLayer).reshape(-1, 1)]

        # for each layer n calculate the output and add it to the array
        for current_layer in range(len(self.weights)):
            layerOut = np.dot(self.weights[curernt_layer], prevLayer[curernt_layer])
            layerOut += self.bias[curernt_layer]
            layerOut = np.vectorize(self.activationFunc)(layerOut)

            prevLayer.append(layerOut)

        # - Backpropagation
        # calulate error for the last layer and adjust weights and biasses
        target = np.array(target_arr).reshape(-1,1)

        # result/output of the prediction is prevLayer[-1]
        current_error = target - prevLayer[-1]
        
        # gradient descent..
        gradient = np.vectorize(self.activationFuncD)(prevLayer[-1])
        gradient = np.multiply(gradient, current_error)
        gradient *= self.learningRate

        # delta for last layer
        prevOut_T = prevLayer[-2].T
        delta = np.dot(gradient, prevOut_T)

        self.bias[-1] += gradient
        self.weights[-1] += delta

        # for each layer n, starting with the 2nd to last, going backwards:
        for current_layer in range(len(self.weights)-2, -1, -1):

            # calculate error (from previous layer to this one)
            current_weight_T = self.weights[curernt_layer+1].transpose()
            current_error = np.dot(current_weight_T, current_error)

            # calculate gradient descent
            gradient = np.vectorize(self.activationFuncD)(prevLayer[curernt_layer+1])
            gradient = np.multiply(gradient, current_error)
            gradient *= self.learningRate

            # delta
            prevOut_T = prevLayer[curernt_layer].T
            delta = np.dot(gradient, prevOut_T)

            self.bias[curernt_layer] += gradient
            self.weights[curernt_layer]  += delta


# - Example test: XOR (non-linear)

"""
nn = NeuralNetwork(2,3,4,1)

train = [
    [[0,0],[0]], 
    [[0,1],[1]], 
    [[1,0],[1]], 
    [[1,1],[0]]
]
x = 0

print("Initial predictions:")
print(f"0 {nn.predict([0,0])}")
print(f"1 {nn.predict([0,1])}")
print(f"1 {nn.predict([1,0])}")
print(f"0 {nn.predict([1,1])}")
while x < 50000:
    x += 1
    if x % 1000 == 0:
        print("Loop at:", x)
    ix = np.random.randint(0,4)
    nn.train(train[ix][0], train[ix][1])

print(f"0 {nn.predict([0,0])}")
print(f"1 {nn.predict([0,1])}")
print(f"1 {nn.predict([1,0])}")
print(f"0 {nn.predict([1,1])}")
"""
