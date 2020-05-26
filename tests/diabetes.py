from DMML.NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np

# normalize data
data = pd.read_csv('tests/diabetes_data.csv').apply(lambda x: x/x.max(), axis=0)

# split data
train = data.sample(frac=0.8)
test = data.sample(frac=0.2)

# print(test)

print("- NeuralNetwork")
nn = NeuralNetwork(8, 80, 1)

print("\n TRAIN \n")
for ix, row in train.iterrows():
   nn.train([row[0:-1]], [row.Outcome])
print("\n TEST \n")

avg = 0.5
for ix, row in test.iterrows():
    result = nn.predict([row[0:-1]])
    error = abs(row.Outcome - result)
    print(result, "<>", row.Outcome)
    avg = (avg + error) / 2

print("avarage Err:", avg)
