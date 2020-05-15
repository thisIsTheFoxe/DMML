import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(y):
    return y * (1-y)

class NeuralNetwork:
    def __init__(self, noInputLayer, noHiddenLayer, noOutputLayer):
        self.weights_ih = np.random.uniform(-1, 1, (noHiddenLayer, noInputLayer))
        self.weights_ho = np.random.uniform(-1, 1, (noOutputLayer, noHiddenLayer))
        self.bias_h = np.random.uniform(-1, 1, (noHiddenLayer))
        self.bias_o = np.random.uniform(-1, 1, (noOutputLayer))

        self.learningRate = 0.1
        self.activationFunc = sigmoid

        print(self.weights_ih, self.bias_h, self.weights_ho, self.bias_o)

    def predict(self, input_arr):
        inputs = np.array(input_arr)
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = np.vectorize(self.activationFunc)(hidden)

        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = np.vectorize(self.activationFunc)(output)

        return output

    def train(self, input_arr, target_arr):
        # run NN (l1)
        inputs = np.array(input_arr)
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = np.vectorize(self.activationFunc)(hidden)
        # run l2
        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = np.vectorize(self.activationFunc)(output)

        ## calc L2:
        targets = np.array(target_arr).reshape(-1,1)
        output_errors = targets - output

        gradient = np.vectorize(dsigmoid)(output)
        gradient = gradient.dot(output_errors)
        gradient *= self.learningRate
        
        # delta
        hidden_T = hidden.transpose()
        ho_deltas = np.dot(gradient, hidden_T)

        self.bias_o += gradient
        self.weights_ho += ho_deltas

        ## calc L1
        hidden_weight_T = self.weights_ho.transpose()
        hidden_errors = hidden_weight_T.dot(output_errors)

        # hidden gradient
        hidden_gradient = np.vectorize(dsigmoid)(hidden)
        hidden_gradient = hidden_gradient.dot(hidden_errors)
        hidden_gradient *= self.learningRate

        input_T = inputs.transpose()
        hi_deltas = np.dot(hidden_gradient, input_T)

        self.bias_h += hidden_gradient
        self.weights_ih += hi_deltas

        print(f"Train, out={output}; target={targets}")
        print(f"Err = {output_errors}")
        print(f"Deltas = {ho_deltas, hi_deltas}")


nn = NeuralNetwork(2, 2, 1)

train = [
    [[0,1],[1]], 
    [[0,0],[0]], 
    [[1,0],[1]], 
    [[1,1],[0]]
]

x = 0
while x < 1000:
    x += 1
    ix = np.random.randint(0,4)
    nn.train(train[ix][0], train[ix][1])


print(f"0,0: {nn.predict([0,0])}")  
print(f"0,1: {nn.predict([0,1])}")
print(f"1,0: {nn.predict([1,0])}")
print(f"1,1: {nn.predict([1,1])}")
