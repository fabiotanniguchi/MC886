# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB

TRAINING_MODEL_FILE = "year-prediction-msd-train.txt"
TESTING_MODEL_FILE= "year-prediction-msd-test.txt"

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    return plt

data = np.loadtxt(open(TRAINING_MODEL_FILE, 'r'),
                     dtype={'names': ('year', 'timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                      'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                      'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                      'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                      'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                      'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                      'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                      'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'),
                            'formats': (np.integer, np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float)},
                    delimiter=',', skiprows=0)

tam = data.size

df = pd.DataFrame(data, columns=['timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                      'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                      'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                      'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                      'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                      'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                      'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                      'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'])
target = pd.DataFrame(data, columns=["year"])

X = df
y = target["year"]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

estimator = GaussianNB()
title = "Learning Curves"
plot_learning_curve(estimator, title, X, y)

plt.show()

data = np.loadtxt(open(TESTING_MODEL_FILE, 'r'),
                     dtype={'names': ('year', 'timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                      'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                      'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                      'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                      'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                      'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                      'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                      'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'),
                            'formats': (np.integer, np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float)},
                    delimiter=',', skiprows=0)

tam = data.size

df2 = pd.DataFrame(data, columns=['timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                      'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                      'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                      'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                      'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                      'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                      'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                      'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'])
X_predict = df2

y_predict = model.predict(X_predict)
