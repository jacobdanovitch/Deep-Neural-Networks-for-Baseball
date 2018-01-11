import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint

def predict(x, weights):
    return np.dot(x, weights)

def cost(prediction, values):
    m = len(prediction)
    err = np.square(prediction - values).sum()
    return err / (2*m)

def relu(x, deriv=False):
    if not deriv:
        return x * (x > 0)
    else:
        return x>0

def generate_weights(size):
    return np.random.random(size)

def neural_network(x, y, hidden_layers, alpha=0.0001, max_iter=10000, verbose=False, predict_data=[], random_seed=1234):
    np.random.seed(random_seed)
    layer0_weights = generate_weights((len(x[0]), hidden_layers)) #layer 0 -> 1
    layer1_weights = generate_weights((hidden_layers,1)) #layer 1 -> 2

    m = len(y)
    cost_history = []

    for i in range(max_iter):
        layer0_output = relu(predict(x, layer0_weights))
        layer1_output = predict(layer0_output, layer1_weights)

        layer1_deltas = layer1_output - y
        layer0_deltas = np.dot(layer1_deltas, layer1_weights.transpose()) * relu(layer0_output, deriv=True)

        layer1_weights -= alpha * np.dot(layer0_output.transpose(), layer1_deltas)
        layer0_weights -= alpha * np.dot(x.transpose(), layer0_deltas)

        cost_history.append(cost(layer1_output, y))

        if i % 10**(len(str(max_iter))-2) == 0 and (i == 0 or i >= 10**(len(str(max_iter))-2)) and verbose:
            print("Iteration", i)
            print("Cost:", cost_history[-1],end="\n \n")

    print("Iteration", max_iter)
    print("Cost:", cost_history[-1])

    if len(predict_data) > 0:
        layer0_output = relu(predict(predict_data, layer0_weights))
        layer1_output = predict(layer0_output, layer1_weights)
        return layer1_output

    model = {
        'L0':layer0_weights.tolist(),
        'L1':layer1_weights.tolist()
    }

    if verbose:
        pprint.pprint(model)

    return model


def main():
    data = pd.read_csv('TestFIPData.csv')
    train_data = data.sample(frac=0.7)
    test_data = data.loc[~data.index.isin(train_data.index)]

    x_train = train_data[['K9', 'BB9', 'HR9']]
    x_test = test_data[['K9', 'BB9', 'HR9']]

    y_train = train_data[['ERA']]
    y_test = test_data[['ERA']]

    '''
    test_df = pd.concat([x_test, y_test], axis=1)
    test_df.loc[:,'gdFIP'] = predict(test_df.iloc[:,:3], weights)
    print(test_df.head(10))
    '''

    data.loc[:,'nnERA'] = neural_network(x_train.values, y_train.values, 4, verbose=True,
                                                 alpha=0.0000001, predict_data=data[['K9', 'BB9', 'HR9']])

    writer = pd.ExcelWriter('nnERA.xlsx')
    data.to_excel(writer,'nnERA')
    writer.save()

main()



