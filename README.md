# Data Mining & Machine Learning
This is a project for the DMML lecture in the 6th semester of the Informatics degree at the Baden-Wuerttemberg Cooperative State University (DHBW) Stuttgart. The project was part of multiple home-work projects during the COVID-pandemic in 2020. 

## Implementation of a Multi-Layer-Perceptron (MLP)
A multi-layer perceeptron is basically a fully connected neural network with at least 3 layers: input, hidden and output. There may be multiple hidden layer and layer can have multiple nodes.

![figure of a 3 layer network](https://miro.medium.com/max/1400/1*-IPQlOd46dlsutIbUq1Zcw.png)

### Usage

1. Install all dependecies  
`pip install -r requirements.txt`
2. Import the neural network  
`from DMML.NeuralNetwork import NeuralNetwork`
3. Create a new neural network  
`nn = NeralNetwork(<numberOfInputNodes>, <numberOfHiddenNodesLayer1>, ..., <numberOfOuputNodes>)`
4. (optional) adjust learning rate and activation function  
```
nn.learningRate = <0-1>
nn.activationFunc = <some function>
nn.activationFuncD = <derivative of that function>
```
5. Train the network  
`nn.train([input1, ...], [output1, ...])`
6. Let the network predict  
`prediction = nn.predict([input1, ...])`