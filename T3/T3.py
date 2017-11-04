# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os


# Activates Verbose on all models.
DEBUG = 0

# Name of the dataset directory.
DATASET_PATH = 'documents/data.csv'



def load_model_data(num_rows = 0, num_features = 0, ignore_features=[]):
    features = np.loadtxt(open(DATASET_PATH, 'rb'), delimiter=',', skiprows=1)
    if (num_rows != 0):
        features = features[range(num_rows)]

    if (num_features != 0):
        features = features[:, range(num_features)]

    for i in range(len(ignore_features)):
        features = np.delete(features, i, axis=1)

    #features = pd.DataFrame(data).abs()
    return features


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

###
# Plots the comparison between the predicted and the actual value.
###
def plot_model_values(values):
    fig, ax = plt.subplots()
    y = list(range(len(values)))
    cmap = get_cmap(len(values[0]))
    for i in range(len(values[0])):
        ax.scatter(values[:,i], y, c = cmap(i))
    ax.legend(range(len(values[0])))
    ax.grid(True)
    plt.show()


###
# Plots the loss curve of the MLP model.
###
def plot_mpl_loss_curve(mlp):
    fig, ax = plt.subplots()
    ax.set_title("Neural Network Loss Rate")
    ax.plot(mlp.loss_curve_, c='red', linestyle='-')
    plt.show()


###
# This function prints and plots the confusion matrix.
###
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    np.set_printoptions(precision=2)
    plt.figure()

    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


###
# Calculates the accuracy of a model.
###
def accuracy(y_test, y_pred):
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count += 1
    acc = 100 * count / len(y_pred)
    return acc


###
# Main function, executes the model prediction.
###
def main():
    features =  load_model_data(100, 10, [0, 1, 2, 3, 4, 5])
    plot_model_values(features)
    #y_pred = run_logistic_regression_onevsall(x_train, y_train, x_test, y_test)
    #y_pred = run_logistic_regression_softmax(x_train, y_train, x_test, y_test)
    #y_pred = run_simple_neural_network_model(x_train, y_train, x_test, y_test)
    #y_pred = run_complex_neural_network_model(x_train, y_train, x_test, y_test)

    #print("Plotting Confusion Matrix...")
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #plot_confusion_matrix(cnf_matrix, classes=labels)

###
# Sets logging information and calls main function.
###
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
