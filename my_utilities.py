import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]) # .predict(X) returns shape(m,), Class labels for samples in X
    cool_print(asd=np.c_[xx.ravel(), yy.ravel()], Z=Z, pprint=False)
    Z = Z.reshape(xx.shape) # Z is svc's prediction for a a bunch of random x and y inputs

    plt.contourf(xx, yy, Z, cmap=plt.cm.hot, alpha=0.2)
    plot_data(X, y)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

def plot_xy(x, **Y):
    plt.xlabel('x')
    plt.ylabel('y')
    for name, y in Y.items():
        plt.plot(x, y, label=name)
    plt.legend(loc='upper right')
    plt.show()

def plot_yx(y, **X):
    plt.xlabel('x')
    plt.ylabel('y')
    for name, x in X.items():
        plt.plot(x, y, label=name)
    plt.legend(loc='upper right')
    plt.show()

def plot_data(X, y):
    x1 = X[:,0]
    x2 = X[:,1]
    colors = []
    markers = []
    for value in y:
        if value == 1:
            colors.append('blue')
            markers.append('o')
        else:
            colors.append('red')
            markers.append('x')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x1, x2, c=colors)
    plt.show()

def sigmoid(z):
    return 1. / (1. + np.exp(-1. * z))  # shape: (5000, 1)

def sigmoid_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))

def cool_print(**kwargs):
    if kwargs['pprint'] is True:
        for key, value in kwargs.items():
            if key != 'pprint':
                if hasattr(value, 'shape'):
                    print("{} shape {}".format(key, value.shape))
                else:
                    print("{} shape {}".format(key, len(value)))
                pprint(value)
    else:
        for key, value in kwargs.items():
            if key != 'pprint':
                if hasattr(value, 'shape'):
                    print("{} shape {}".format(key, value.shape))
                else:
                    print("{} shape {}".format(key, len(value)))