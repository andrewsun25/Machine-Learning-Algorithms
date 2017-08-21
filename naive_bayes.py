import numpy as np
import pandas as pd
import scipy.stats as st
from pprint import pprint
import my_utilities as my

print(st.norm.pdf(0.))
"""
Naive Bayes Classifier- Generative algorithm not Discriminative
Discriminative- learns conditional probability distribution p(y|X)
Generative- learns join probability distribution p(X, y)

Naive assumptions- 
1. features are independent of each other. p(X, y) = p(x1 | y) * p(x2 | y) * .. * p(xn | y) * p(y)
2. each example is equally likely p(X1) = p(X2) = .. = p(Xm)
3. if features are correlated, naive bayes can't detect it. Assumes covariance matrix with covariances of 0

Central Idea- For each given input X, predict Y = y, where y is the class that is most probable given the input X.
p(y | X) = p(x1 | y) * p(x2 | y) * .. * p(xn | y) * p(y)

Algorithm-
1. Using training set, compute the parameters (IE mean, std) for each feature for each class y
For Y == y1:
    find mean, std of (x1...xn)
For Y == y2:
    find mean, std of (x1...xn)
...
For Y == yc:
    find mean, std of (x1...xn)
2. Using test set, given each example Xm, find the most probable class y
    h1 = argmax(
        p(Y=y1 | X1) = p(x1 | y1),parametrized by mean(x1 | y1), std(x1 | y1) * p(x2 | y1) * .. * p(xn | y) * p(y)
        ...
        p(Y=yc | X1)
    )
    ...
    hm = argmax(p(Y | Xm))
note: Each feature might follow a different distribution
"""

def train_params(data, size_train):
    # y_train shape (515, 1)
    y_train = np.vstack(data.iloc[:size_train, -1].values)
    
    # X_train_negative shape (330, 8)
    X_negative = []
    y_negative_total = 0
    for i, y in np.ndenumerate(y_train):
        if y == 0:
            X_negative.append(data.iloc[i[0], :len(data.columns) - 1])
            y_negative_total += 1
    X_negative = pd.DataFrame(X_negative)
    # # p( y == 0)
    y_negative_prior = 1. * y_negative_total / y_train.shape[0]
    # X_negative_means shape (8,)
    X_negative_means = np.array([
        np.mean(X_negative[col]) for col in X_negative
    ])
    # X_negative_std shape (8,)
    X_negative_std = np.array([
        np.std(X_negative[col]) for col in X_negative
    ])
    
    # X_positive shape (185, 8)
    X_positive = []
    for i, y in np.ndenumerate(y_train):
        if y == 1:
            X_positive.append(data.iloc[i[0], :len(data.columns) - 1])
    X_positive = pd.DataFrame(X_positive)
    # p( y == 1)
    y_positive_prior = 1. - y_negative_prior
    # X_positive_means shape (8,)
    X_positive_means = np.array([
        np.mean(X_positive[col]) for col in X_positive
    ])
    # X_positive_std shape (8,)
    X_positive_std = np.array([
        np.std(X_positive[col]) for col in X_positive
    ])
    
    trained_params = {
        'means_negative': X_negative_means,
        'means_positive': X_positive_means,
        'std_negative': X_negative_std,
        'std_positive': X_positive_std,
        'prior_negative': y_negative_prior,
        'prior_positive': y_positive_prior
    }
    return trained_params

def predict(trained_params, X_test):
    p_y_negative = []
    p_y_positive = []
    h = []
    for example in X_test:
        likelihood_negative = 1
        likelihood_positive = 1
        for i, feature in enumerate(example):
            likelihood_negative *= st.norm.pdf(feature, loc=trained_params['means_negative'][i], scale=trained_params['std_negative'][i])
            likelihood_positive *= st.norm.pdf(feature, loc=trained_params['means_positive'][i], scale=trained_params['std_positive'][i])
        likelihood_negative *= trained_params['prior_negative']
        likelihood_positive *= trained_params['prior_positive']
        p_y_negative.append(likelihood_negative)
        p_y_positive.append(likelihood_positive)
    for i, p_negative in enumerate(p_y_negative):
        if p_negative > p_y_positive[i]:
            h.append(0)
        else:
            h.append(1)
    return np.array(h)

def compare(h, y_test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, prediction in enumerate(h):
        if prediction == y_test[i] and prediction == 1:
            # print("TP !! H: {}, Y: {}".format(prediction, y_test[i]))
            TP += 1
        elif prediction == y_test[i] and prediction == 0:
            # print("TN !! H: {}, Y: {}".format(prediction, y_test[i]))
            TN += 1
        elif prediction != y_test[i] and prediction == 1:
            # print("FP !! H: {}, Y: {}".format(prediction, y_test[i]))
            FP += 1
        else:
            # print("FN !! H: {}, Y: {}".format(prediction, y_test[i]))
            FN += 1
    m = len(y_test)
    accuracy = 1. * (TP + TN) / m
    recall = 1. * TP / (TP + FN)
    precision = 1. * TP / (TP + FP)
    print("Accuracy: {} Recall: {} Precision: {}".format(accuracy, recall, precision))

### Execution
data = pd.read_csv('data/pima-indians-diabetes.csv', header=None) # shape (767, 9)
# size_train 515
prop_train = 0.67
size_train = int(round(prop_train * data.shape[0]))
trained_params = train_params(data, size_train)
# X_test shape (253, 8)
X_test = data.iloc[size_train:, :-1].values
# y_test shape (253, 1)
y_test = data.iloc[size_train:, -1].values
h = predict(trained_params, X_test)
compare(h, y_test)


