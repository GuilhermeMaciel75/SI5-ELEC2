import numpy as np
from math import sqrt
from random import randrange
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = np.sum((row1[:-1] - row2[:-1])**2)
    return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
    distances = np.array([euclidean_distance(codebook, test_row) for codebook in codebooks])
    return codebooks[np.argmin(distances)]

# Make a prediction with codebook vectors
def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    return bmu[-1]

# Create a random codebook vector
def random_codebook(train):
    n_records = len(train)
    n_features = train.shape[1]
    codebook = train[randrange(n_records), :]
    return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = np.array([random_codebook(train) for _ in range(n_codebooks)])
    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    return codebooks

# LVQ Algorithm as a custom classifier
class LVQClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_codebooks=10, lrate=0.1, epochs=100):
        self.n_codebooks = n_codebooks
        self.lrate = lrate
        self.epochs = epochs
    
    def fit(self, X, y):
        # Validate the input arrays
        X = check_array(X)
        y = np.array(y)

        # Combine features and labels
        train = np.column_stack((X, y))
        self.codebooks_ = train_codebooks(train, self.n_codebooks, self.lrate, self.epochs)
        return self

    def predict(self, X):
        X = check_array(X)
        predictions = np.array([predict(self.codebooks_, row) for row in X])
        return predictions
    
    def predict_proba(self, X):
        X = check_array(X)
        probabilities = []
        for row in X:
            bmu = get_best_matching_unit(self.codebooks_, row)
            class_label = bmu[-1]
            # For simplicity, we will return a binary classification probability for each class
            prob = np.array([0.0, 0.0])  # Assuming binary classification
            prob[class_label] = 1.0  # Fully confident in the class of the BMU
            probabilities.append(prob)
        return np.array(probabilities)

    def decision_function(self, X):
        X = check_array(X)
        # This function will return the distances to the codebooks, which can be used for decision making
        distances = np.array([euclidean_distance(get_best_matching_unit(self.codebooks_, row), row) for row in X])
        return distances
