# Data Mining & Machine Learning
This is a project for the DMML lecture in the 6th semester of the Informatics degree at the Baden-Wuerttemberg Cooperative State University (DHBW) Stuttgart. The project was part of multiple home-work projects during the COVID-pandemic in 2020. 

## Implementation of a Multi-Layer-Perceptron (MLP)
A multi-layer perceeptron is basically a fully connected neural network with at least 3 layers: input, hidden and output. There may be multiple hidden layers and each layer can have multiple nodes.

![figure of a 3 layer network](https://miro.medium.com/max/1400/1*-IPQlOd46dlsutIbUq1Zcw.png)

### Usage

1. Install all dependecies  
`pip install -r requirements.txt`
2. Import the neural network  
`from DMML.NeuralNetwork import NeuralNetwork`
3. Create a new neural network  
`nn = NeuralNetwork(<numberOfInputNodes>, <numberOfHiddenNodesLayer1>, ..., <numberOfOuputNodes>)`
4. (optional) adjust learning rate and activation function, defaults to 0.1 and Sigmoid.
```
nn.learningRate = <0-1>
nn.activationFunc = <some function>
nn.activationFuncD = <derivative of that function>
```
5. Train the network  
`nn.train([input1, ...], [output1, ...])`
6. Let the network predict  
`prediction = nn.predict([input1, ...])`

### `ThreeLayerNeuralNetwork`
This is a neural network with exactly 3 layers (exactly 1 hidden layer). It was the first, simple implementation of a MLP. The functionality is very similar to the new `NeuralNetwork`. Therefore the general `NeuralNetwork` should be used instead. 

## CI
There are two CI tests. The first, trivial test just prooves that the CI is working as expected. The second test creates a singlelayer perceptron that solves the linear seperable OR-problem. 

![Python application](https://github.com/thisIsTheFoxe/DMML/workflows/Python%20application/badge.svg)

## Testing the Multi-Layer-Perceptron
The tests can be found in DMML/tests.

### Non-linear Functions
For this test we define two functions: 
1. a circle
2. a default quadratic function

Now, we pick 50.000 random points were x and y are between -1 and 1. The network should be trained to recognize if points are in a certain area, defined by the functions. The first output indicates whether the point is within the "Unit Circle" (scaled by 0.5). The second function shows if the point lies below plot of the quadratic function. (Output relate to boolean values: 0 = false; 1 = true)

After that we again pick 10 random points and see how well out network performs. Example:
```
[0.99977432] [0.90243369] <> 1 1
[0.0151541] [0.99999997] <> 0 1
[0.01278248] [0.65513176] <> 0 1
[0.05145783] [0.02938048] <> 0 0
[0.08897768] [0.00031185] <> 0 0
[0.00601403] [0.04847992] <> 0 0
[0.0259609] [0.99819635] <> 0 1
[0.99974666] [0.01196247] <> 1 0
[0.99987619] [0.52585137] <> 1 0
[0.99996185] [0.44944496] <> 1 0
```
Left we see the output of the network and right the actual result. Apart from the second to last, all values look good. Intuitively, (0 0) means the point is not in the circle and above the quadratic plot and (1 1) means the point is in the circle and below the quadratic plot. 

![plot of both functions](/resources/function.png)

### Iris Dataset
The Iris dataset is a popular dataset of Iris flowers. It contains measurements from 150 flowers of 3 different species. 

To work with that dataset we first need to split it into training and test data. So we split out data according to the 3 species. Then we shuffle each of the three. Then we take 5 samples from each one for out test data and shuffle everything in test and trainig data together.

One problem in this case is, that not all the data is equally prioritized. E.g. sepalLength and sepalWidth have much higher values and therefore are seen by the neural network as 'more important'. 

![wikipedia plot of the iris set](/resources/iris.png)

### Diabetes
This is a dataset about people with diabetes. When looking at the data though, one can recognize that only about ⅓ of the people tested positive for diabetes. Plus, the data is unclean meaning not all the data is trustworthy (for example: BloodPressure = 0, BMI = 0). For this reason, the network doesn't perform as well with the unsanitized data. 


## Resources
[3B1B - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
[CodingTrain/Toy-Neural-Network-JS](https://github.com/CodingTrain/Toy-Neural-Network-JS)
[Iris Dataset](https://forge.scilab.org/index.php/p/rdataset/source/tree/master/csv/datasets/iris.csv)
