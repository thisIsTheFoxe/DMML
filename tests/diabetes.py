from DMML.perceptron import NeuralNetwork
import pandas as pd
import numpy as np

# bad/dirty data
data = pd.read_csv('tests/diabetes_data.csv')


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

train, _, test = train_validate_test_split(data, .8, 0)

# print(train.loc[1][0:-1])

print("- NeuralNetwork")
nn = NeuralNetwork(8, 4, 10, 1)

print("\n TRAIN \n")
for ix, row in train.iterrows():
   nn.train([row[0:-1]], [row.Outcome])
print("\n TEST \n")

avg = 0.5
for ix, row in test.iterrows():
    # print(row)
    result = nn.predict([row[0:-1]])
    error = abs(row.Outcome - result)
    print(result, error)
    avg = (avg + error) / 2

print("avarage Err:", avg)
