import numpy as np
import pandas as pd


class Layer:
    def __init__(self, position, max_position, alpha):
        """
        :param position: Layer's position in the overall network
        :param max_position: Network's width
        :param alpha: Learning rate
        """
        self.position = position
        self.max_position = max_position
        self.alpha = alpha

        self.weights = []  # weights of layer
        self.prediction = None  # output of layer
        self.delta = None  # delta for backprop weight adjustment
        self.cost = None  # cost (only used for final output layer)

        print("Layer",position,"initialized.\n")

    def generate_weights(self, size):
        self.weights = np.random.random(size)  # randomly initialize weights given tuple
        print("Layer", self.position,"weights:",self.weights,'\n ')  # print for confirmation

    def get_weights(self):
        return self.weights

    def update_weights(self, delta):
        self.weights -= self.alpha * delta

    def relu(self, x, deriv=False):
        if not deriv:
            return x * (x > 0)  # Sets output to 0 if input is negative
        else:
            return x > 0  # Output is 1 if x is positive

    def predict(self, x, return_pred=False):
        if self.position == self.max_position:
            # Output layer prediction; no relu activation
            self.prediction = np.dot(x, self.weights)
        else:
            # Hidden layer predictions; use relu activation
            self.prediction = self.relu(np.dot(x, self.weights))

        if return_pred:
            return self.prediction

    def get_prediction(self):
        return self.prediction

    def calculate_cost(self, prediction, values):
        m = len(prediction)
        err = np.square(prediction - values).sum()
        return err / (2 * m)

    def get_cost(self):
        return self.cost

    def get_delta(self):
        return self.delta

    def calculate_delta(self, y=None, prev_layer=None):
        if self.position == self.max_position:
            # only for final output layer
            self.delta = self.prediction - y  # subtracts y/truth vector from prediction vector
            self.cost = self.calculate_cost(self.prediction, y.values)  # adds cost
        else:
            """
            delta calculation for backpropagating through hidden layers
            
            dot product of last layer's delta and transposed weights, 
            multiplied by relu derivative to freeze negative weights 
            """
            self.delta = np.dot(prev_layer.get_delta(), prev_layer.get_weights().transpose()) * self.relu(self.prediction,
                                                                                                          deriv=True)


class NeuralNetwork:
    def __init__(self, num_layers, x, y, max_iter=1000, alpha=0.01, random_seed=1234):
        '''
        HYPERPARAMS
        :param num_layers: total layers in network
        :param max_iter: iterations for gradient descent
        :param alpha: learning rate (passed to layers)
        :param random_seed: random seed for weight generation (Change this if your network sucks!!)
        '''

        self.alpha = alpha

        self.num_layers = num_layers
        self.layers = []  # list containing all Layer objects

        self.max_iter = max_iter
        self.random_seed = random_seed

        '''
        DATA
        :param x: input vector/matrix
        :param features: shape of input / # of features
        :param y: ground truth vector/matrix
        '''

        self.x = x
        self.features = len(x.iloc[0])
        self.y = y

    def init_layers(self):
        np.random.seed(self.random_seed)  # for deterministic weight generation
        max_position = self.num_layers - 1  # width of network

        # null initialize layers
        self.layers = [Layer(i, max_position, alpha=self.alpha) for i in range(self.num_layers)]

        # Generate weights for first layer (will always be as wide as # of features)
        self.layers[0].generate_weights((self.features, self.num_layers))

        # If user specifies for more hidden layers
        if self.num_layers > 1:
            # Generate weights for middle hidden layers (won't run for num_layers == 2)
            for layer in self.layers[1:max_position]:
                # Need to make number of neurons editable
                layer.generate_weights((self.num_layers, self.num_layers))

            # Generate weight for output layer (should have single output for regression)
            self.layers[-1].generate_weights((self.num_layers, 1))

    def forward_prop(self):
        pred = self.layers[0].predict(self.x, return_pred=True) # initial prediction from input vector
        for l, layer in enumerate(self.layers[1:]):
            # update prediction iterating through rest of layers, chaining prediction
            pred = layer.predict(pred, return_pred=True)

    def back_prop(self):
        for l, layer in reversed(list(enumerate(self.layers))):
            # for final output layer
            if l == len(self.layers) - 1:
                # calculate initial delta for backprop
                layer.calculate_delta(y=self.y)
            # for rest of backprop
            else:
                # use previous layer to backprop and calculate delta
                layer.calculate_delta(prev_layer=self.layers[l + 1])

    def update_weights(self):
        for l, layer in enumerate(self.layers):
            # weight update for first layer (multiplied by alpha inside function)
            if l == 0:
                # input vector transposed, dotted with input vector's delta
                layer.update_weights(np.dot(self.x.transpose(), layer.get_delta()))
            else:
                # uses previous layer to update weights
                prev_layer = self.layers[l - 1]
                layer.update_weights(np.dot(prev_layer.get_prediction().transpose(), layer.get_delta()))

    def train(self):
        # initialize layers
        self.init_layers()

        for epoch in range(1,self.max_iter+1):
            # forward, back, update
            self.forward_prop()
            self.back_prop()
            self.update_weights()

            # print every thousand
            if epoch % 1000 == 0:
                self.print_layers(epoch)

    def print_layers(self,epoch=-1):
        if epoch >= 0:
            print("Epoch",epoch,"complete.")

        for l, layer in enumerate(self.layers):
            print("Layer",l)
            print(layer.get_weights())
            print()

        print("Cost:", self.layers[-1].get_cost())

    def fit(self, x):
        # this is broke ngl
        pred = self.layers[0].predict(x)
        for l, layer in enumerate(self.layers[1:]):
            pred = layer.predict(pred, return_pred=True)
        return pred


def main():
    """
    This is just project-specific stuff
    """
    data = pd.read_csv('TestFIPData.csv')
    train_data = data.sample(frac=0.7)
    test_data = data.loc[~data.index.isin(train_data.index)]

    x_train = train_data[['K9', 'BB9', 'HR9']]
    x_test = test_data[['K9', 'BB9', 'HR9']]

    y_train = train_data[['ERA']]
    y_test = test_data[['ERA']]

    neural_net = NeuralNetwork(3, x_train, y_train, max_iter=20000, alpha=0.0000001)
    neural_net.train()

    '''
    this doesn't work yet
    predictions = neural_net.fit(x_test)
    print(predictions)
    '''

    '''
    boiler plate code for saving to xlsx
    
    test_df = pd.concat([x_test, y_test], axis=1)
    test_df.loc[:,'gdFIP'] = predict(test_df.iloc[:,:3], weights)
    print(test_df.head(10))
    

    data.loc[:,'nnERA'] = neural_network(x_train.values, y_train.values, 4, verbose=True,
                                                 alpha=0.0000001, predict_data=data[['K9', 'BB9', 'HR9']])

    writer = pd.ExcelWriter('nnERA.xlsx')
    data.to_excel(writer,'nnERA')
    writer.save()
    '''


# execute
main()
