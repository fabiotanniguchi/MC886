# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics.pairwise import pairwise_distances


# Activates Verbose on all models.
DEBUG = 0

# Name of the dataset directory.
DATASET_PATH = 'documents/data.csv'

# CSV FILE HAS 19924 ROWS AND 2209 COLUMNS
# EACH ROW REPRESENTS A DOCUMENT

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

def load_model_data(num_rows = 0, num_features = 0, ignore_features=[]):
    features = np.loadtxt(open(DATASET_PATH, 'rb'), delimiter=',', skiprows=1)
    
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
    features = load_model_data(100, 10, [0, 1, 2, 3, 4])
    #plot_model_values(features)
    
    # distance matrix
    d = pairwise_distances(features, metric='euclidean')
    
    # split into 2 clusters
    M, C = kMedoids(d, 10)

    print('medoids:')
    for point_idx in M:
        print( features[point_idx] )

    print('')
    print('clustering result:')
    for label in C:
        for point_idx in C[label]:
            print('label {0}:　{1}'.format(label, features[point_idx]))
    
    
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
