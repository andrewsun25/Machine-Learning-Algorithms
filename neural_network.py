from my_utilities import *
from scipy.io import loadmat
from scipy import optimize
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint


class NeuralNetwork:

    def _X_y(self):
        data = loadmat('data/ex4data1.mat')
        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(data['y'])
        return data['X'], y_onehot

    def _layer_units(self):
        layer_units = []
        layer_units.append(self.n)
        for j in range(0, self.hidden):
            layer_units.append(self.k)
        layer_units.append(self.c)
        return np.array(layer_units)

    def _random_weights(self):
        weights = []
        for i, prev_units in enumerate(self.layer_units[:-1]):
            weights.append(
                np.random.random((self.layer_units[i + 1], 1 + prev_units))
            )
        return np.array(weights)

    # n input features, k units in hidden layer, number of hidden layers, c output classes,
    def __init__(self, n=400, k=25, hidden=2, c=10, activation=sigmoid):
        self.n = n
        self.k = k
        self.hidden = hidden
        self.num_layers = hidden + 2
        self.c = c
        self.layer_units = self._layer_units() # np.array([n, k, k, c]
        self.weights = self._random_weights()
        self.X, self.y = self._X_y()
        self.activation = activation

    def reroll(self, theta_unrolled, **kwargs):
        total_layers = self.hidden + 2
        for key, value in kwargs.items():
            theta0 = np.reshape(theta_unrolled[:10426], (26, 401))
        theta1 = np.reshape(theta_unrolled[10426:], (10, 26))
        return theta0, theta1

    def combine_flatten(matrix0, matrix1):
        return np.concatenate((matrix0.flatten(), matrix1.flatten()))

    def prop_forward(self, i):
        a0 = np.insert(
            np.vstack(self.X[i]), 0, np.array([1]), axis=0
        )
        a = [a0]
        z = []
        for i in range(0, self.num_layers - 1):
            if i != self.num_layers - 2:
                z.append(self.weights[i].dot(a[i]))
                a.append(
                    np.insert(
                        sigmoid(z[i]), 0, np.array([1]), axis=0
                    )
                )
            else:
                z.append(self.weights[i].dot(a[i]))
                a.append(
                    sigmoid(z[i])
                )
        return np.array(a)

    def prop_backward(self, i):
        a = self.prop_forward(i)
        y_i = np.vstack(self.y[i])
        d_output = a[-1] - y_i
        d = [d_output]
        for j in range(self.num_layers - 1, 0, -1):
            if j <= self.hidden:
                weights_biased = np.insert(self.weights[j - 1], 0, 1, axis=0)
                d.insert(0, weights_biased.T.dot(d[0]) * (a[j - 1] * (1 - a[j - 1])))
            else:
                d.insert( 0, self.weights[j-1].T.dot(d[0]) * (a[j-1] * (1 - a[j-1])) )
        d.insert(0,[])
        return np.array(d)

    def cost_i(y_i, h):
        return y_i.T.dot(np.log(h)) + (1 - y_i).T.dot(1 - np.log(h))  # shape (1,1)

    def cost_gradient_wrapper(theta_unrolled, X, y):
        theta0, theta1 = reroll(theta_unrolled)
        delta0 = np.zeros((26, 401))
        delta1 = np.zeros((10, 26))
        cost = 0
        for i in range(0, 5000):
            a0 = np.vstack(X[i])
            y_i = np.vstack(y[i])
            a1, a2 = prop_forward(theta0, theta1, a0)
            cost += np.sum(cost_i(y_i, a2))
            d1, d2 = prop_backward(theta0, theta1, a0, a1, a2, y_i)
            delta0 += d1.dot(a0.T)
            delta1 += d2.dot(a1.T)
            # cool_print(a0=a0, y_i=y_i, cost=cost, a1=a1, a2=a2, d1=d1, d2=d2, delta0=delta0, delta1=delta1, pprint=True)
        cost_gradient_0 = 1. / X.shape[0] * delta0
        cost_gradient_1 = 1. / X.shape[0] * delta1
        cost_gradient = combine_flatten(cost_gradient_0, cost_gradient_1)
        print(cost)
        return cost, cost_gradient


