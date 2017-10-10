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
import matplotlib.pyplot as plt
import logging
import os


# VALIDATION TESTING
# NUM_TRAIN = 400000
# NUM_VAL   = 100000
# NUM_TEST  = 0

# FINAL TESTING ONLY.
NUM_TRAIN = 50000
NUM_VAL   = 0
NUM_TEST  = 10000

# Logistic Regression parameters.
LOGISTIC_TOLERANCE = 0.0003
LOGISTIC_C = 1
LOGISTIC_MAX_ITER = 100
LOGISTIC_JOBS = 4

# ANN parameters.
ANN_MAX_ITER = 30
ANN_ACTIVATION = 'relu'

# Activates Verbose on all models.
DEBUG = 0

# Name of the cifar directory.
DATASET_PATH = 'cifar-10-batches-py'


###
# Auxilliary function that reads a single CIFAR file.
###
def load_CIFAR_batch(filename):
    with open(filename, 'r') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 1024).transpose(0, 2, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR_labels(filename):
    with open(filename, 'r') as f:
        datadict = pickle.load(f)
        labels = datadict['label_names']
        labels = np.array(labels)
        return labels


###
# Auxilliary function to load the CIFAR files.
###
def load(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    labels = load_CIFAR_labels(os.path.join(ROOT, 'batches.meta'))
    return Xtr, Ytr, Xte, Yte, labels


###
# Loads data from CIFAR, extracting the training, validation and test data.
###
def get_CIFAR10_data(cifar10_dir, num_training=49000, num_validation=1000, num_test=1000):
    '''
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the neural net classifier.
    '''
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test, labels = load(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]


    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Transpose so that channels come first
    # X_train = X_train.transpose(0, 3, 1, 2)
    # X_val = X_val.transpose(0, 3, 1, 2)
    # X_test = X_test.transpose(0, 3, 1, 2)

    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train)

    # Data Normalization.
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_train /= std
    X_val /= std
    X_test /= std

    # Normalize between 0 and 1.
    # maxv = np.max(X_train, axis=0)
    # X_train /= maxv
    # X_val /= maxv
    # X_test /= maxv

    X_train = X_train.reshape((num_training, 3072))
    X_val = X_val.reshape((num_validation, 3072))
    X_test = X_test.reshape((num_test, 3072))

    return {
        "x_train": X_train, "y_train": y_train,
        "x_val": X_val, "y_val": y_val,
        "x_test": X_test, "y_test": y_test,
        "mean": mean_image, "std": std,
        "labels": labels
    }


###
# Plots the comparison between the predicted and the actual value.
###
def plot_model_values(x_values, y_values):
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, edgecolors = (0, 0, 0))
    ax.legend()
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
# Run logistic regression one-vs-all
###
def run_logistic_regression_onevsall(x_train, y_train, x_test, y_test):
    print("------------------------------------------")
    print("------LOGISTIC REGRESSION ONE-VS-ALL------")

    print("Training...")
    logreg = LogisticRegression(tol=LOGISTIC_TOLERANCE, C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, verbose=DEBUG, multi_class='ovr', n_jobs=LOGISTIC_JOBS, solver='saga')
    logreg.fit(x_train, y_train)

    print("Testing...")
    test_predict = logreg.predict(x_test)

    print('Test Accuracy score: %.2f%%' % (accuracy(y_test, test_predict)))
    print("--------------------||--------------------")
    return test_predict


###
# Run logistic regression using SOFTMAX
###
def run_logistic_regression_softmax(x_train, y_train, x_test, y_test):
    print("------------------------------------------")
    print("-----LOGISTIC REGRESSION USING SOFTMAX----")
    print("Training...")
    logreg = LogisticRegression(tol=LOGISTIC_TOLERANCE, C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, verbose=DEBUG, multi_class='multinomial', solver='saga')
    logreg.fit(x_train, y_train)

    print("Testing...")
    test_prob = logreg.predict_proba(x_test)
    test_predict = a = np.empty(len(y_test), dtype=int)
    for x in range(len(test_prob)):
        maxi = 0
        maxval = 0
        for i in range(len(test_prob[x])):
            if(test_prob[x][i] > maxval):
                maxi = i
                maxval = test_prob[x][i]
        test_predict[x] = maxi

    print('Test Accuracy score: %.2f%%' % (accuracy(y_test, test_predict)))
    print("--------------------||--------------------")
    return test_predict


###
# Runs a simple neural network with only one hidden layer.
###
def run_simple_neural_network_model(x_train, y_train, x_test, y_test):
    print("------------------------------------------")
    print("---------ARTIFICIAL NEURAL NETWORK--------")

    print("Training...")
    print("Number of elements on hidden layer: %d" % (len(x_train[0])/3))
    model = MLPClassifier(activation=ANN_ACTIVATION, max_iter=ANN_MAX_ITER, solver='sgd', verbose=DEBUG, hidden_layer_sizes=(len(3))
    model.fit(x_train, y_train)
    print("Testing...")
    test_predict = model.predict(x_test)

    print("Training set loss: %f" % model.loss_)
    print('Test Accuracy score: %.2f%%' % (accuracy(y_test, test_predict)))

    print("Plotting Loss Curve...")
    #plot_mpl_loss_curve(model)

    print("--------------------||--------------------")
    return test_predict


###
# Runs a more complex neural network with two hidden layer.
###
def run_complex_neural_network_model(x_train, y_train, x_test, y_test):
    print("------------------------------------------")
    print("--------ARTIFICIAL NEURAL NETWORK 2-------")

    print("Training...")
    layer_size = len(x_train[0])/3
    print("Number of elements on hidden layer: %d" % (layer_size))
    model = MLPClassifier(activation=ANN_ACTIVATION, max_iter=ANN_MAX_ITER, solver='sgd', verbose=DEBUG, hidden_layer_sizes=(layer_size, layer_size))
    model.fit(x_train, y_train)
    print("Testing...")
    test_predict = model.predict(x_test)

    print("Training set loss: %f" % model.loss_)
    print('Test Accuracy score: %.2f%%' % (accuracy(y_test, test_predict)))

    print("Plotting Loss Curve...")
    #plot_mpl_loss_curve(model)

    print("--------------------||--------------------")
    return test_predict


###
# Main function, executes the model prediction.
###
def main():
    data =  get_CIFAR10_data(DATASET_PATH,
                            num_training=NUM_TRAIN, num_validation=NUM_VAL, num_test=NUM_TEST)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test  = data['x_test']
    y_test  = data['y_test']
    labels  = data['labels']

    # Not using test file, assume validation.
    if (len(y_test) == 0):
        x_test  = data['x_val']
        y_test  = data['y_val']

    y_pred = run_logistic_regression_onevsall(x_train, y_train, x_test, y_test)
    #y_pred = run_logistic_regression_softmax(x_train, y_train, x_test, y_test)
    #y_pred = run_simple_neural_network_model(x_train, y_train, x_test, y_test)
    #y_pred = run_complex_neural_network_model(x_train, y_train, x_test, y_test)

    print("Plotting Confusion Matrix...")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=labels)

###
# Sets logging information and calls main function.
###
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
