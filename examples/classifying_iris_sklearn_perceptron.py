# source: https://www.pycodemates.com/2022/12/perceptron-algorithm-understanding-and-implementation-python.html

from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Loading the dataset
iris = load_iris()

# Splitting the dataset
X = iris.data[:, (0, 1)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

sk_perceptron = Perceptron()
sk_perceptron.fit(X_train, y_train)
sk_perceptron_pred = sk_perceptron.predict(X_test)

# Accuracy
accuracy_score(sk_perceptron_pred, y_test)
