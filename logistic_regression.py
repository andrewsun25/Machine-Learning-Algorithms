import numpy as np
from scipy.io import loadmat
from scipy import optimize
from pprint import pprint

def calc_g(theta, X):
    z = X.dot(theta)
    return 1. / (1. + np.exp(-z)) # shape: (5000, 1)


def calc_j(theta, X, y, l=0.):
    theta = np.vstack(theta)
    term1 = -1. * y.T.dot( np.log( calc_g(theta, X) ) )
    term2 = 1. * (1 - y).T.dot( np.log( 1 - calc_g(theta, X) ) )
    reg_term = l / 2. * theta.T.dot(theta)
    #print("J shape", (1. / X.shape[0] * ( (term1 - term2) + reg_term )).shape)
    return np.sum((1. / X.shape[0] * ( (term1 - term2) + reg_term ))) # shape(1, 1)

def calc_grad(theta, X, y, l=0.):
    # print(1. / X.shape[0] * ( X.T.dot( calc_g(theta, X) - y ) + l * theta ).shape)
    # print("Grad shape", (1. / X.shape[0] * ( X.T.dot( calc_g(theta, X) - y ) + l * theta )).shape)
    theta = np.vstack(theta)
    list = (1. / X.shape[0] * ( X.T.dot( calc_g(theta, X) - y ) + l * theta )).ravel()
    return list
    # return 1. / X.shape[0] * ( X.T.dot( calc_g(theta, X) - y ) + l * theta ) # shape: (400, 1)

def predict(X, y, k, l=0.):
    all_theta = np.zeros((k, X.shape[1]))
    theta_0 = np.zeros(X.shape[1])
    for i in range(1, k+1):
        # Converts y_i into one hot vector
        y_i = np.array([1 if number == i else 0 for number in y]).reshape(y.shape[0], 1)
        # Fills all_theta with optimized values
        all_theta[i-1] = optimize.minimize(fun=calc_j, x0=theta_0, args=(X, y_i, l), method='TNC', jac=calc_grad).x
    all_h = calc_g(all_theta.T, X)
    print(all_h.shape)
    print(all_h)
    h_argmax = np.argmax(all_h, axis=1)
    h_argmax = h_argmax + 1
    return h_argmax



data = loadmat('../data/ex3data1.mat')
X = data['X']
y = data['y']
y_pred = predict(X, y, 10)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))



