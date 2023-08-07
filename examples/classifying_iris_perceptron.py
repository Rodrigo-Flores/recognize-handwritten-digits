# source: https://www.pycodemates.com/2022/12/perceptron-algorithm-understanding-and-implementation-python.html

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import numpy as np
from ..Perceptron import Perceptron

# Loading the dataset
iris = load_iris()

# Splitting the dataset
X = iris.data[:, (0, 1)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

# Training and making predictions
perceptron = Perceptron(0.001, 100)

perceptron.fit(X_train, y_train)

pred = perceptron.predict(X_test)

# Now let's see how much accuracy we have got,
accuracy_score(pred, y_test)

# Classification report
report = classification_report(pred, y_test, digits=2)
print(report)
