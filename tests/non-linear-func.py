from DMML.perceptron import NeuralNetwork
import numpy as np

def isInCircle(x, y):
    return x**2+y**2 <= 0.5

def isBelowCubed(x,y):
    return y <= x**2

nn = NeuralNetwork(2, 50, 2)

for i in range(50000):
    if i % 1000 == 0: print("i =", i)
    x,y = np.random.uniform(-1, 1, 2)
    cir = 1 if isInCircle(x,y) else 0
    cub = 1 if isBelowCubed(x,y) else 0
    nn.train([x, y], [cir, cub])

for i in range(10):
    x,y = np.random.uniform(-1, 1, 2)
    cir = 1 if isInCircle(x,y) else 0
    cub = 1 if isBelowCubed(x,y) else 0

    pCir, pCub = nn.predict([x,y])
    print(pCir, pCub, "<>", cir, cub)

