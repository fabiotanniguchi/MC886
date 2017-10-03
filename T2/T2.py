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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import pprint

TRAINING_MODEL_FILE = "year-prediction-msd-train.txt"
TESTING_MODEL_FILE = "year-prediction-msd-test.txt"
USE_TEST_FILE = False


###
# Loads the training data, returning X and Y.
###
def load_model_data(test_file = TRAINING_MODEL_FILE):
    data = np.loadtxt(open(test_file, 'r'),
                         dtype={'names': ('year',
                                          'timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                          'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                          'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                          'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                          'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                          'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                          'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                          'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'
                                          ),
                                'formats': (np.integer, np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float)},
                        delimiter=',', skiprows=0)

    df = pd.DataFrame(data, columns=[
                              'timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                          'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                          'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                          'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                          'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                          'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                          'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                          'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'
                            ]).abs()
    target = pd.DataFrame(data, columns=["year"])
    scaler = preprocessing.StandardScaler(with_mean=False).fit(df)
    return df, target, scaler.transform(df), target["year"].tolist()


###
# Loads the training and testing data.
###
def load_training_testing_data(X, Y):
    # By default load the test data by splitting the training data.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0, shuffle=True)
    if (True == USE_TEST_FILE):
        # Load the test file instead.
        df, target, x_test, y_test = load_model_data(TESTING_MODEL_FILE)
        x_train = X
        y_train = Y
    return x_train, x_test, y_train, y_test


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


def predict_linear_model(x_train, y_train, x_test, y_test):
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    print("Linear model:", pretty_print_linear(model.coef_))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors = (0, 0, 0), color = 'red')
    ax.plot([1920, 2020], [1920, 2020], 'k--', lw = 4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")

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



def predict_linear_model_polinomial(x_train, y_train, x_test, y_test):
    lm = linear_model.LinearRegression()
    model = make_pipeline(PolynomialFeatures(), lm)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors = (0, 0, 0), color = 'red')
    ax.plot([1920, 2020], [1920, 2020], 'k--', lw = 4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
      model, x_train, y_train, train_sizes=[0.2, 0.5, 0.7, 1], cv=4)
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


###
# Main function, executes the model prediction.
###
def main():
    data, target, X, Y = load_model_data()
    x_train, x_test, y_train, y_test = load_training_testing_data(X, Y)

    #plot_model_values(x_test, y_test)
    #predict_linear_model(x_train, y_train, x_test, y_test)
    predict_sgd_model(x_train, y_train, x_test, y_test)
    #predict_linear_model_polinomial(x_train, y_train, x_test, y_test)


###
# Sets logging information and calls main function.
###
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
