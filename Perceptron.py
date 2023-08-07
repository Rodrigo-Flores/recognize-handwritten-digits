# source: https://www.pycodemates.com/2022/12/perceptron-algorithm-understanding-and-implementation-python.html

import numpy as np

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    # heaviside activation function
    def activation(self, z):
        """
        0 if z < 0
        1 if  >= 0
        """
        return np.heaviside(z, 0)  # haviside(z) heaviside -> activation

    def fit(self, X, y):
        n_features = X.shape[1]

        # Initializing weights and bias
        self.weights = np.zeros((n_features))
        self.bias = 0

        # Iterating until the number of epochs
        for epoch in range(self.epochs):

            # Traversing through the entire training set
            for i in range(len(X)):
                # Finding the dot product and adding the bias
                z = np.dot(X, self.weights) + self.bias
                # Passing through an activation function
                y_pred = self.activation(z)

                # Updating weights and bias
                self.weights = self.weights + \
                    self.learning_rate * (y[i] - y_pred[i]) * X[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])

        return self.weights, self.bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
