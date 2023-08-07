import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from Perceptron import Perceptron

# Create a random seed for reproducibility
np.random.seed(42)

# Create two classes with two features
# Class 0: Mean at (2, 2) and standard deviation 0.5
# Class 1: Mean at (5, 5) and standard deviation 0.5
num_samples_per_class = 100
class_0 = np.random.normal(
    loc=[2, 2], scale=0.5, size=(num_samples_per_class, 2))
class_1 = np.random.normal(
    loc=[5, 5], scale=0.5, size=(num_samples_per_class, 2))

# Combine the two classes
X = np.vstack((class_0, class_1))
y = np.hstack((np.zeros(num_samples_per_class),
              np.ones(num_samples_per_class)))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

# Training and making predictions
perceptron = Perceptron(0.001, 100)
perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)

# Now let's see how much accuracy we have got,
accuracy = accuracy_score(pred, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
report = classification_report(pred, y_test, digits=2)
print(report)
