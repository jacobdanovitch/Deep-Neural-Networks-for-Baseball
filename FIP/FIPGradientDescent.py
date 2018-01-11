import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x, weights):
    return np.dot(x, weights)

def cost(prediction, values):
    m = len(prediction)
    err = np.square(prediction - values).sum()
    return err / (2*m)

def gradient_descent(x, y, alpha=0.0001, max_iter=10000, random_seed=1234, verbose=False):
    np.random.seed(random_seed)
    weights = np.random.random((len(x[0]), 1))

    m = len(y)
    cost_history = []

    for i in range(max_iter):
        pred = predict(x, weights)
        delta = pred - y

        weight_deltas = np.dot(x.transpose(), delta)
        weights -= alpha*weight_deltas

        '''
        for i in range(len(x)):
            pred = np.dot(x[i], weights)
            delta = (pred - y[i])
            weight_deltas = alpha*np.array([delta*input for input in x[i]])
            weights = weights - weight_deltas
        '''

        cost_history.append(cost(pred, y))

        if i % 100 == 0 and verbose:
            print("Iteration", i)
            print("Cost:", cost_history[-1],end="\n \n")

    print("Iteration", max_iter)
    print("Cost:", cost_history[-1])

    return [i[0] for i in weights]

data = pd.read_csv('TestFIPData.csv')
train_data = data.sample(frac=0.7)
test_data = data.loc[~data.index.isin(train_data.index)]

x_train = train_data[['K9', 'BB9', 'HR9']]
x_test = test_data[['K9', 'BB9', 'HR9']]

y_train = train_data[['ERA']]
y_test = test_data[['ERA']]

weights = gradient_descent(x_train.values, y_train.values, alpha=0.000001)

'''
test_df = pd.concat([x_test, y_test], axis=1)
test_df.loc[:,'gdFIP'] = predict(test_df.iloc[:,:3], weights)
print(test_df.head(10))
'''

data.loc[:,'gdFIP'] = predict(data[['K9', 'BB9', 'HR9']], weights)

writer = pd.ExcelWriter('validation_rows.xlsx')
data.to_excel(writer,'gdFIP')
writer.save()


