import numpy as np
import itertools
from scipy.io import loadmat
from scipy import optimize
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint


def cool_print(**kwargs):
    if kwargs['pprint'] == True:
        for key, value in kwargs.items():
            if key != 'pprint':
                print("{} shape {}".format(key, value.shape))
                pprint(value)
    else:
        for key, value in kwargs.items():
            if key != 'pprint':
                print("{} shape {}".format(key, value.shape))

def random_weights():
    return np.random.random((26, 401)), np.random.random((10, 26))

def X_y():
    data = loadmat('../data/ex4data1.mat')
    encoder = OneHotEncoder(sparse=False)
    X = np.concatenate((
        np.ones((5000, 1)), data['X']
    ), axis=1)
    y_onehot = encoder.fit_transform(data['y'])
    return X, y_onehot

def sigmoid(z):
    return 1. / (1. + np.exp(-z)) # shape: (5000, 1)

def sigmoid_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))

def prop_forward(theta0, theta1, a0):
    z1 = theta0.dot(a0)
    a1 = sigmoid(z1) # shape (26, 1)
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2) # shape (10, 1) Output
    return a1, a2

def prop_backward(theta0, theta1, a0, a1, a2, y_i):
    d2 = a2 - y_i # (10, 1), error of output
    d1 = theta1.T.dot(d2) * (a1 * (1 - a1))
    return d1, d2

def cost_i(y_i, h):
    return y_i.T.dot( np.log(h) ) + (1 - y_i).T.dot( 1 - np.log(h) ) # shape (1,1)

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
        cool_print(a0=a0, y_i=y_i, cost=cost, a1=a1, a2=a2, d1=d1, d2=d2, delta0=delta0, delta1=delta1, pprint=True)
    cost_gradient_0 = 1. / X.shape[0] * delta0
    cost_gradient_1 = 1. / X.shape[0] * delta1
    cost_gradient = combine_flatten(cost_gradient_0, cost_gradient_1)
    cool_print(cost_gradient=cost_gradient, pprint=True)
    print(cost)
    return cost, cost_gradient

def reroll(theta_unrolled):
    theta0 = np.reshape(theta_unrolled[:10426], (26, 401))
    theta1 = np.reshape(theta_unrolled[10426:], (10, 26))
    return theta0, theta1

def combine_flatten(matrix0, matrix1):
    return np.concatenate((matrix0.flatten(), matrix1.flatten()))


# class Trainer:
#     def __init__(self):
#         self.costs = []
#         self.X, self.y = X_y()
#         self.theta0, self.theta1 = random_weights()
#         self.theta_unrolled = combine_flatten(self.theta0, self.theta1)
#         self.options = {
#             'maxiter': 50,
#             'disp': True
#         }
#     def optimize(self):
#         theta_optimal_unrolled = optimize.minimize(fun=cost_gradient_wrapper, x0=self.theta_unrolled, args=(self.X, self.y), method='TNC', jac=True, options=self.options).x
#         theta0_optimal, theta1_optimal = reroll(theta_optimal_unrolled)
#         cool_print(theta0_optimal=theta0_optimal, theta1_optimal=theta1_optimal, theta_optimal_unrolled=theta_optimal_unrolled, pprint=True)

X, y = X_y()
theta0, theta1 = random_weights()
theta_unrolled = combine_flatten(theta0, theta1)
options = {
'maxiter': 50,
'disp': True
}
theta_optimal_unrolled = optimize.minimize(fun=cost_gradient_wrapper, x0=theta_unrolled, args=(X, y), method='TNC', jac=True, options=options).x
theta0_optimal, theta1_optimal = reroll(theta_optimal_unrolled)
cool_print(theta0_optimal=theta0_optimal, theta1_optimal=theta1_optimal, theta_optimal_unrolled=theta_optimal_unrolled, pprint=True)

# theta_unrolled = combine_flatten(theta0, theta1)
# theta0, theta1 = reroll(theta_unrolled)
# cool_print(theta0=theta0, theta1=theta1, pprint=True)
# theta_unrolled = np.ravel(theta_big[0].flatten(), theta_big[1].flatten())
# cost_gradient_wrapper(X, y)

#optimal_theta = optimize.minimize(fun=cost, x0=theta_unrolled, args=(X, y), method='TNC', jac=cost_gradient).x


# a0 shape (401, 1)
# y_i shape (10, 1)
# cost shape ()
# a1 shape (26, 1)
# a2 shape (10, 1)
# d1 shape (26, 1)
# d2 shape (10, 1)
# delta0 shape (26, 401)
# delta1 shape (10, 26)

