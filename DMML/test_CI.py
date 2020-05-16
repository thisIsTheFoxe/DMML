from NeuralNetwork import NeuralNetwork

OR = [
    [[0,0],[0]], 
    [[0,1],[1]], 
    [[1,0],[1]], 
    [[1,1],[1]]
]

# test if CI runs this test
def inc(x):
    return x + 1


def test_answer():
    assert inc(4) == 5

def test_SLP():
    nn = NeuralNetwork(2, 1, 1)

    x = 0
    while x < 1000:
        x += 1
        nn.train(OR[0][0], OR[0][1])
        nn.train(OR[1][0], OR[1][1])
        nn.train(OR[2][0], OR[2][1])
        nn.train(OR[3][0], OR[3][1])
    
    pred0 = nn.predict([0,0])
    pred1 = nn.predict([0,1])
    pred2 = nn.predict([1,0])
    pred3 = nn.predict([1,1])

    assert pred0 < pred1
    assert pred0 < pred2
    assert pred0 < pred3

test_SLP()