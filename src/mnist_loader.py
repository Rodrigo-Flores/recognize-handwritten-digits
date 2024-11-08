import pickle
import gzip
import numpy as np

def load_data() -> tuple:
  """
  Load the MNIST data from the file.

  Desc:
    Load the MNIST data from the file `mnist.pkl.gz`. The file is a gzipped
    pickle file for the MNIST dataset.

  Args:
    None

  Returns:
    tuple: A tuple containing the training data, validation data, and test data.
  """
  with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
  return (training_data, validation_data, test_data)

def load_data_wrapper() -> tuple:
  """
  Load the MNIST data from the file.

  Desc:
    Load the MNIST data from the file `mnist.pkl.gz`. The file is a gzipped
    pickle file for the MNIST dataset.

  Args:
    None

  Returns:
    tuple: A tuple containing the training data, validation data, and test data.
  """
  tr_d, va_d, te_d = load_data()
  training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = list(zip(training_inputs, training_results))
  validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
  validation_data = list(zip(validation_inputs, va_d[1]))
  test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
  test_data = list(zip(test_inputs, te_d[1]))
  return (training_data, validation_data, test_data)

def vectorized_result(j) -> np.ndarray:
  """
  Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.

  Desc:
    Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.
    This is used to convert a digit (0...9) into a corresponding desired output from the neural network.

  Args:
    j (int): The digit to be converted.

  Returns:
    np.ndarray: The 10-dimensional unit vector.
  """
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e