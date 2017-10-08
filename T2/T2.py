# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

NSIZE = 3
N_STARTING_FILTERS = 16

NUM_PROCESSES = 4

NUM_TRAIN = 10000
NUM_TEST = 1000

DATASET_PATH = 'cifar-10-batches-py'
USE_TEST_FILE = False

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
    # X_train = X_train.transpose(0, 3, 1, 2)
    # X_val = X_val.transpose(0, 3, 1, 2)
    # X_test = X_test.transpose(0, 3, 1, 2)

    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train /= std
    X_val /= std
    X_test /= std

    X_train = X_train.reshape((num_training, 3072))
    #X_val = X_train.reshape((num_validation, 3072))
    X_test = X_test.reshape((num_test, 3072))

    return {
        "x_train": X_train, "y_train": y_train,
        "x_val": X_val, "y_val": y_val,
        "x_test": X_test, "y_test": y_test,
        "mean": mean_image, "std": std
    }


def load_CIFAR_batch(filename):
    ''' load single batch of cifar '''
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 1024).transpose(0, 2, 1).astype("float")
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
def load_data():
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
    model = MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(len(x_train[0])),random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    #print("Linear model:", pretty_print_linear(model.coef_))

    count = 0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            count=count+1
    acc = count / len(y_pred)

    print('Accuracy score: %.2f' % (acc))

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
    
###
# Run logistic regression one-vs-all
###
def run_logistic_regression_onevsall(x_train, y_train, x_test, y_test):
    print("--------------------------------------")
    print("----LOGISTIC REGRESSION ONE-VS-ALL----")
    
    accuracy_array = []
    
    for c in range(0, 10):
        y_train_c = []
        for x in range(0, len(y_train)):
            if(y_train[x] == c):
                y_train_c.append(1)
            else:
                y_train_c.append(0)
        
        logreg = LogisticRegression('l2', False, 0.0001, 1.0, True, 1, None, None, 'saga', 100, 'ovr', 1, True, -1)
        logreg.fit(x_train, y_train_c)
        
        test_predict = logreg.predict(x_test)
        acertos = 0
        for x in range(0, len(test_predict)):
            if(test_predict[x] == 1):
                if(c== y_test[x]):
                    acertos += 1
            if(test_predict[x] == 0):
                if(c != y_test[x]):
                    acertos += 1
        
        my_accuracy = 100.0 * acertos / len(test_predict)
        accuracy_array.append(my_accuracy)
    
    accuracy = sum(accuracy_array) / len(accuracy_array)    
    print ('Testing accuracy: {}%'.format(accuracy))
    print("----------------||--------------------")

###
# Run logistic regression using SOFTMAX
###
def run_logistic_regression_softmax(x_train, y_train, x_test, y_test):
    print("-----------------------------------------")
    print("----LOGISTIC REGRESSION USING SOFTMAX----")
    print("Training...")
    logreg = LogisticRegression('l2', False, 0.0001, 1.0, True, 1, None, None, 'saga', 100, 'multinomial', 1, True, -1)
    logreg.fit(x_train, y_train)

    print("Testing...")
    test_predict = logreg.predict_proba(x_test)
      
    acertos = 0
    for x in range(0, len(test_predict)):
        x_probas = test_predict[x]
        max_proba_idx = np.argmax(x_probas)
        if max_proba_idx == y_test[x]:
            acertos += 1
  
    my_accuracy = 100.0 * acertos / len(test_predict)
    print ('Testing accuracy: {}%'.format(my_accuracy))
    print("------------------||---------------------")

###
# Main function, executes the model prediction.
###
def main():
    data = load_data()
    #plot_model_values(data['x_test'], data['y_test'])
    #predict_neural_model(data['x_train'], data['y_train'], data['x_test'], data['y_test'])

    # Dados estão no formato (50000 linhas, 3072 colunas), com cada coluna sendo um valor de pixel.
    # Se quiser separar em 3 features de cada um dos canais é só comentar as linhas 75 e 77.
    
    run_logistic_regression_onevsall(data['x_train'], data['y_train'], data['x_test'], data['y_test'])
    run_logistic_regression_softmax(data['x_train'], data['y_train'], data['x_test'], data['y_test'])


###
# Sets logging information and calls main function.
###
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
