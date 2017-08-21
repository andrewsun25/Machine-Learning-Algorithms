import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from pprint import pprint
# Class methods take in a class as the first argument not an object.
# 1. Start at (t1=0, t2=0)


class Display:

    @staticmethod
    def plotData(df):
        square_feet = df['square_feet']
        bedrooms = df['bedrooms']
        price = df['price']
        plt.xlabel('fields')
        plt.ylabel('price')
        plt.legend()
        plt.scatter(square_feet, price, c='red')
        plt.scatter(bedrooms, price, c='blue')
        plt.show()

    @staticmethod
    def plot3D(df, weights, y):
        square_feet = df['square_feet']
        bedrooms = df['bedrooms']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('square_feet')
        ax.set_ylabel('bedrooms')
        ax.set_zlabel('predicted price')
        ax.scatter(square_feet, bedrooms, y)
        #X, Y = np.meshgrid(square_feet, bedrooms)
        def fun(x, y):
            return weights[0] + weights[1] * x + weights[2] * y
        Z = np.array([fun(square_feet, bedrooms) for x in square_feet])
        ax.plot_surface(square_feet, square_feet, Z)
        plt.show()

    @staticmethod
    def plotIterationCost(iterations, costs):
        plt.xlabel('# Iterations')
        plt.ylabel('cost')
        plt.plot(iterations, costs)
        plt.show()


class LinearRegression:

    def __init__(self, data_raw, alpha):
        self._square_feet = (data_raw.iloc[:, 0] - np.mean(data_raw.iloc[:, 0])) / np.std(data_raw.iloc[:, 0])
        self._bedrooms = (data_raw.iloc[:, 1] - np.mean(data_raw.iloc[:, 1])) / np.std(data_raw.iloc[:, 1])
        self.prices = (data_raw.iloc[:, 2] - np.mean(data_raw.iloc[:, 2])) / np.std(data_raw.iloc[:, 2])
        x_df = pd.DataFrame({
            'field_0': 1,
            'square_feet': self._square_feet,
            'bedrooms': self._bedrooms,
        })
        self.x_df = x_df[['field_0', 'square_feet', 'bedrooms']]
        self.n = len(self.x_df.iloc[:,0])
        self.m = len(self._square_feet)
        self.weights = np.zeros([1, self.n])
        self.alpha = alpha

    def iterate(self, N):
        iterations = []
        costs = []
        final_h = []
        for iteration in range(0, N):
            cost = self.calcCost()
            self.setNewWeights()
            iterations.append(iteration)
            costs.append(cost)
        for i, value in enumerate(self.x_df.iloc[:,1]):
            final_h.append( self.calcH( self.weights, self.calcXVector(i) ) )
        Display.plotIterationCost(iterations, costs)
        Display.plot3D(self.x_df, self.weights[0], self.prices)

    def setNewWeights(self):
        # Adds new Weights calculated by Gradient Descent to weights_temp array so we can set all weights at the same time
        weights_temp = []
        for j in range(0, self.n):
            weights_temp.append(self.calcNewWeight(j))
        # Actually sets the Weights
        for j, weight in enumerate(weights_temp):
            self.weights[0, j] = weight

    def calcNewWeight(self, j):
        return self.weights[0, j] - self.alpha * self.calcGradient(j)

    def calcGradient(self, j):
        gradient_total = 0
        for i in range(0, self.m):
            x_vector = self.calcXVector(i)
            h = self.calcH(self.weights, x_vector)
            y = self.prices[i]
            gradient_total += (h - y) * x_vector[j, 0]
        gradient_average = gradient_total / self.m
        return gradient_average

    # Returns [[x1...xn]] vector for given row i
    def calcXVector(self, i):
        x_vector = []
        for j in range(0, self.n):
            column_j = self.x_df.iloc[:, j]
            x_vector.append(column_j[i])
        x_vector = np.matrix(x_vector)
        x_vector = x_vector.T
        return x_vector

    # Returns J(theta)
    def calcCost(self):
        weights = self.weights
        error_total = 0
        for i in range(0, self.m):
            x_vector = self.calcXVector(i)
            h = self.calcH(weights, x_vector)
            y = self.prices[i]
            error_squared = (h - y) ** 2
            error_total += error_squared
        cost = error_total / (2 * self.m)
        return cost

    # Returns h(x)
    def calcH(self, weights_matrix, x_vector):
        h = weights_matrix.dot(x_vector)
        return h[0, 0]

data_raw = pd.read_csv('../data/ex1data2.csv')
lr = LinearRegression(data_raw, alpha=1.1)
lr.iterate(30)
