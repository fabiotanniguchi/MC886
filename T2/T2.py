# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import pprint

NSIZE = 3
N_STARTING_FILTERS = 16

NUM_PROCESSES = 4

NUM_TRAIN = 50000
NUM_TEST = 10000

DATA_PATH = '../cifar-10-batches-py'
USE_TEST_FILE = False

path_set = False
while not path_set:
    with open(DATA_PATH) as f:
        DATASET_PATH = f.read()
    path_set = True


def get_CIFAR10_data(cifar10_dir, num_training=49000, num_validation=1000, num_test=1000):
    '''
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the neural net classifier.
    '''
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load(cifar10_dir)

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
    X_train = X_train.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train /= std
    X_val /= std
    X_test /= std

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'mean': mean_image, 'std': std
    }


def load_CIFAR_batch(filename):
    ''' load single batch of cifar '''
    with open(filename, 'r') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load(ROOT):
    ''' load all of cifar '''
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
    return Xtr, Ytr, Xte, Yte

###
# Loads the training and testing data, returning X and Y.
###
def load_data(test_file = TRAINING_MODEL_FILE):
    data = get_CIFAR10_data(DATASET_PATH,
                            num_training=NUM_TRAIN, num_validation=0, num_test=NUM_TEST)
    return data


#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


def plot_model_values(x_values, y_values):
    fig, ax = plt.subplots()
    xis = [[[] for i in range(len(x_values))] for i in range(len(x_values[0]))]
    for xi in range(len(x_values[0])):
        for i in range(len(x_values)):
            xis[xi][i] = x_values[i][xi]
    for i in range(len(xis)):
        ax.scatter(xis[i], y_values, edgecolors = (0, 0, 0), label="X%d" % (i))
    ax.set_xlabel('Variables')
    ax.set_ylabel('Year')
    ax.legend()
    ax.grid(True)
    plt.show()


def predict_neural_model(x_train, y_train, x_test, y_test):
    model = MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    #print("Linear model:", pretty_print_linear(model.coef_))

    count = 0
    for i in range(len(y_pred)):
        if pred[i]==a[i]:
            count=count+1

    print('Accuracy score: %.2f' % count / len(y_pred))

    # fig, ax = plt.subplots()
    # ax.scatter(y_test, y_pred, edgecolors = (0, 0, 0), color = 'red')
    # ax.plot([1920, 2020], [1920, 2020], 'k--', lw = 4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()
    #
    # plt.figure()
    # plt.xlabel("Training examples")
    # plt.ylabel("Score")

    # train_sizes, train_scores, test_scores = learning_curve(
    #   model, x_train, y_train, train_sizes=[0.2, 0.5, 0.7], cv=4)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)

    # plt.grid()
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
    #          label="Training score")
    # plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
    #          label="Cross-validation score")

    # plt.legend(loc="best")
    # plt.show()


def predict_sgd_model(x_train, y_train, x_test, y_test):
    iterations = [500]
    score = [0, 0, 0, 0, 0]
    for index, ite in enumerate(iterations):
        sgd = linear_model.SGDRegressor(alpha=0.0001, average=False, epsilon=0.01, eta0=0.001,
           fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
           loss='squared_loss', max_iter=ite, penalty='l2',
           power_t=0.35, random_state=None, shuffle=True, tol=None,
           verbose=0, warm_start=False)
        model = sgd.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Linear model:", pretty_print_linear(model.coef_))
        score[index] = r2_score(y_test, y_pred)
        train_sizes, train_scores, test_scores = learning_curve(
          model, x_train, y_train, train_sizes=[0.2, 0.5, 0.7], cv=4)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

    # fig, ax = plt.subplots()
    # plt.plot(iterations, score, 'o-', color="r",
    #          label="Training score")
    # plt.xlabel("Iterations")
    # plt.ylabel("Score")
    # plt.show()


###
# Main function, executes the model prediction.
###
def main():
    data = load_data()
    plot_model_values(x_test, y_test)
    predict_neural_model(data['x_train'], data['y_train'], data['x_test'], data['y_test'])
    predict_sgd_model(data['x_train'], data['y_train'], data['x_test'], data['y_test'])


###
# Sets logging information and calls main function.
###
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
