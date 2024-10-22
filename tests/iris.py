from DMML.NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np

#import data
data = pd.read_csv('tests/iris.csv')

#split data
dA = data[:50].sample(frac=1)
dB = data[50:100].sample(frac=1)
dC = data[100:].sample(frac=1)

trainingData = pd.concat([dA[5:],dB[5:],dC[5:]]).sample(frac=1)
testData = pd.concat([dA[:5],dB[:5],dC[:5]]).sample(frac=1)

# print(testData)

def formatResult(name):
    if name == "Setosa": return [1,0,0]
    elif name == "Versicolor": return [0,1,0]
    elif name == "Virginica": return [0,0,1]
    else: return [0,0,0]


print("- NeuralNetwork")
nn = NeuralNetwork(4, 10, 50, 3)

print("\n TRAIN \n")
for ix, row in trainingData.iterrows():
    input = row[0:-1].apply(pd.to_numeric, errors='coerce')
    nn.train([input], formatResult(row[-1]))
print("\n TEST \n")

for ix, row in testData.iterrows():
    result = nn.predict([row[0:-1]])
    print(result, "<>", formatResult(row[-1]))
