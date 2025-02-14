import numpy as np
from warnings import warn

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(y):
    return y * (1-y)


# DEPRECATED! Use NeuralNetwork instead
# A NN with one input, one hidden and one output layer.
class ThreeLayerNeuralNetwork:

    def __init__(self, noInputLayer, noHiddenLayer, noOutputLayer):

        warn("`ThreeLayerNeuralNetwork` is deprecated. Please use `NeuralNetwork` instead.")
        # initiaize weights as (toNode, fromNode) matrices
        self.weights_ih = np.random.uniform(-1, 1, (noHiddenLayer, noInputLayer))
        self.weights_ho = np.random.uniform(-1, 1, (noOutputLayer, noHiddenLayer))

        # random.uniform creates an array that's handled like a (1,2) matrix
        #  but we want a (2,1) one.. so, reshape
        self.bias_h = np.random.uniform(-1, 1, (noHiddenLayer)).reshape(-1, 1)
        self.bias_o = np.random.uniform(-1, 1, (noOutputLayer)).reshape(-1, 1)

        # default reaning rate
        self.learningRate = 0.1

        # set activation function & it's d/dx
        self.activationFunc = sigmoid
        self.activationFuncD = dsigmoid

        print(self.weights_ih, self.bias_h, self.weights_ho, self.bias_o)

    def predict(self, input_arr):
        # reshape array to an (n, 1) vertical matrix
        inputs = np.array(input_arr).reshape(-1, 1)

        # run NN l1
        hidden = np.dot(self.weights_ih, inputs)
        hidden += self.bias_h
        hidden = np.vectorize(self.activationFunc)(hidden)

        #run NN l2
        output = np.dot(self.weights_ho, hidden)
        output += self.bias_o
        output = np.vectorize(self.activationFunc)(output)
        return output

    def train(self, input_arr, target_arr):
        # code repetition on purpose to save variables

        # reshape array to an (n, 1) vertical matrix
        inputs = np.array(input_arr).reshape(-1, 1)

        # run NN l1 (fowrward)
        hidden = np.dot(self.weights_ih, inputs)
        hidden += self.bias_h
        hidden = np.vectorize(self.activationFunc)(hidden)

        #run NN l2
        output = np.dot(self.weights_ho, hidden)
        output += self.bias_o
        output = np.vectorize(self.activationFunc)(output)

        # Backpropagation:
        # - calc L2:
        targets = np.array(target_arr).reshape(-1,1)
        output_errors = targets - output

        # gradient descent..
        gradient = np.vectorize(self.activationFuncD)(output)
        gradient = np.multiply(gradient, output_errors)
        gradient *= self.learningRate

        # delta
        hidden_T = hidden.T
        ho_deltas = np.dot(gradient, hidden_T)

        self.bias_o += gradient
        self.weights_ho += ho_deltas

        # - calc L1
        hidden_weight_T = self.weights_ho.transpose()
        hidden_errors = np.dot(hidden_weight_T, output_errors)
        hidden_errors = np.array(hidden_errors)

        # gradient descent..
        hidden_gradient = np.vectorize(self.activationFuncD)(hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient *= self.learningRate

        # delta
        input_T = inputs.T
        ih_deltas = np.dot(hidden_gradient, input_T)

        self.bias_h += hidden_gradient
        self.weights_ih += ih_deltas


# - Example test: XOR (non-linear)

"""
nn = ThreeLayerNeuralNetwork(2,4,1)
nn.train([1,2],[1])

train = [
    [[0,0],[0]], 
    [[0,1],[1]], 
    [[1,0],[1]], 
    [[1,1],[0]]
]

x = 0
while x < 50000:
    x += 1
    if x % 1000 == 0:
        print("Loop at:", x)
    ix = np.random.randint(0,4)
    nn.train(train[ix][0], train[ix][1])

print("FinalNetwork:")
print(nn.weights_ih, nn.bias_h, nn.weights_ho, nn.bias_o)

print(f"0 {nn.predict([0,0])}")
print(f"1 {nn.predict([0,1])}")
print(f"1 {nn.predict([1,0])}")
print(f"0 {nn.predict([1,1])}")
"""