import numpy as np
import pandas as pd
import idx2numpy
from sklearn.metrics import accuracy_score

def get_feature_data(filename):
    initial_frame = idx2numpy.convert_from_file(filename)
    rows = initial_frame.shape[0]
    columns = initial_frame.shape[1] * initial_frame.shape[2]

    # Reshape into a 2D array
    reshaped = np.reshape(initial_frame, (rows, columns))

    # Add a column of ones so that the bias vector can treated as a feature
    bias = np.ones((rows, 1))
    expanded = np.append(reshaped, bias, axis=1)

    # Return in matrix form
    return np.matrix(expanded)

def get_label_data(filename):
    initial_frame = idx2numpy.convert_from_file(filename)
    dummied = pd.get_dummies(initial_frame).to_numpy()
    matrix = np.matrix(dummied)
    # convert to column vector and return
    return matrix.T

def calculate_accuracy(features, weights, labels):
    c_label = pd.DataFrame(labels.T).idxmax(axis=1)
    c_predict = pd.DataFrame((features * weights)).idxmax(axis=1)

    accuracy = accuracy_score(c_label.values, c_predict.values)
    return accuracy