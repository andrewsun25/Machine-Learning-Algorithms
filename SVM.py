from scipy.io import loadmat
import numpy as np
from scipy import optimize
from pprint import pprint
import my_utilities as my
from sklearn import svm
from sklearn.preprocessing import normalize

def k_gauss(x_i, x_j, sigma_squared=5):
    x_diff = x_i - x_j
    return np.exp(-1. * x_diff.T.dot(x_diff) / (2 * sigma_squared) )

def optimal_c(X, y):
    m = len(X) # 863
    size_train = round(.8 * len(X)) # 690


    # test_gamma = 0.01
    c = []
    error_train = []
    error_cv = []
    test_c = 0.00001
    while test_c < 1:
        # Training set
        x_train = X[:size_train, :]
        y_train = y[:size_train]
        # CV set
        x_cv = X[size_train:, :]
        svc = svm.SVC(C=test_c, kernel='rbf')

        # Train on training set
        svc.fit(x_train, y_train)

        # Calculate error on training set
        h_train = svc.predict(x_train)  # returns h, shape(690,)

        error_train_cum = (h_train - y_train).T.dot(h_train - y_train)
        error_train.append(error_train_cum)

        # Calculate error on CV set
        h_cv = svc.predict(x_cv)
        y_cv = y[size_train:]
        error_cv_cum = (h_cv - y_cv).T.dot(h_cv - y_cv)
        error_cv.append(error_cv_cum)

        # test_gamma = test_gamma * 3
        c.append(test_c)
        test_c = test_c * 3
        # h_train = svc.predict(X) # returns h, shape(863-690,)
        #
    my.cool_print(c=c, error_train=error_train, error_cv=error_cv, pprint=False)
    my.plot_xy(c, trainError=error_train, cvError=error_cv)


data = loadmat('data/ex6data2.mat')
# X = normalize(data['X'])
# svc = svm.SVC(C=1, kernel='rbf')
X = data['X'] # shape (863, 2)
y = np.ravel(data['y']) # shape (863,)
x1 = np.vstack(X[:, 0]) # shape (863, 1)
x2 = np.vstack(X[:, 1]) # shape (863, 1)
optimal_c(X, y)
# optimal_c(X)
# cool_print(X=X, y=y, x1=x1, x2=x2, pprint=False)
# plot_data(X, y)
# svc.fit(X, y)
# plot_svc(svc, X, y)


# pprint( svc.score(X, y) )


