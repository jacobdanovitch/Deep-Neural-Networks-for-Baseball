# Deep Neural Networks for Baseball

On October 11, 2017, I made a note in my repository for my other baseball analysis project that said:
> I regret my resentment towards (this project) for being "just a sports stats project." I think I should really try to do more projects like this one.

New Years resolution = complete :tada:

In reading [Andrew Trask's "Grokking Deep Learning"](https://www.manning.com/books/grokking-deep-learning), I've tried to hold myself accountable to learning the material by implementing it from scratch in creative ways. To do so, I've started this project that attempts to use different forms of neural network architecture to predict baseball statistics. Starting basic with predicting a pitcher's ERA using very basic inputs to get my bearings, I aim to eventually progress to much more sophisticated metrics.

## How it works

* `/FIP/FIPGradientDescent.py`: Modelling ERA using basic inputs with simple linear regression by gradient descent
* `/FIP/FIPNN.py`: Modelling ERA using basic inputs with a neural network containing a hidden layer
* `/FIP/FIP-DNN.py`: Modelling ERA using an arbitrarily wide deep neural network

## What I Learned

I learned about different architectures of neural networks, the theories behind learning, and started to grasp the intuition behind backpropagation as credit assignment.

## Future Plans

* Keep learning new models! 
* Use more sophisticated inputs and outputs, nested models

## Current (Known) Problems

* `/FIP/FIP-DNN.py`: NeuralNetwork.fit() doesn't work

### Built With

* **Python** - Pandas, numpy
