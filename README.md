# Data Mining & Machine Learning
## Implementation of a Multi-Layer-Perceptron (MLP)

A Project for DHBW Stuttgart

## CI
There are two CI tests. The first, trivial test just prooves that the CI is working as expected. The second test creates a singlelayer perceptron that solves the linear seperable OR-problem. 

## Testing the Multi-Layer-Perceptron
The tests can be found in DMML/test.

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

### Iris Dataset
The Iris dataset is a popular dataset of Iris flowers. It contains measurements from 150 flowers of 3 different species. 

To work with that dataset we first need to split it into training and test data. So we split out data according to the 3 species. Then we shuffle each of the three. Then we take 5 samples from each one for out test data and shuffle everything in test and trainig data together. Example:
```
.....
```

### Diabetes
This is a dataset about people with diabetis. When looking at the data tho, one can recognize, that only about 1/3 of the people tested have diabetes. Plus, it is unclean meaning not all the data is trustworthy (for some: BloodPresuare = 0, BMI = 0). For this reason, just taking the set, splitting it randomly and working with that, results in a the network classifying almost every patient with a 30% probability of having diabetes. 