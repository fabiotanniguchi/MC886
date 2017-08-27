# -*- coding: utf-8 -*-
"""
asdasdasdas
"""

from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import pandas as pd

TRAINING_MODEL_FILE = "year-prediction-msd-train.txt"
TESTING_MODEL_FILE= "year-prediction-msd-test.txt"

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

df = pd.DataFrame(data, columns=['timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                      'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                      'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                      'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                      'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                      'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                      'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                      'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'])
target = pd.DataFrame(data, columns=["year"])

scaler = preprocessing.StandardScaler().fit(df)

X = scaler.transform(df)
y = target["year"].tolist()

X_sgd = scaler.transform(df)
y_sgd = target["year"].tolist()

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

sgd = linear_model.SGDRegressor()
model_sgd = sgd.fit(X_sgd, y_sgd)

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

df2 = pd.DataFrame(data, columns=['timbre01', 'timbre02', 'timbre03', 'timbre04', 'timbre05', 'timbre06', 'timbre07', 'timbre08', 'timbre09', 'timbre10', 'timbre11', 'timbre12',
                                      'timbrec1','timbrec2','timbrec3','timbrec4','timbrec5','timbrec6','timbrec7','timbrec8','timbrec9','timbrec10','timbrec11','timbrec12',
                                      'timbrec13','timbrec14','timbrec15','timbrec16','timbrec17','timbrec18','timbrec19','timbrec20','timbrec21','timbrec22','timbrec23','timbrec24',
                                      'timbrec25','timbrec26','timbrec27','timbrec28','timbrec29','timbrec30','timbrec31','timbrec32','timbrec33','timbrec34','timbrec35','timbrec36',
                                      'timbrec37','timbrec38','timbrec39','timbrec40','timbrec41','timbrec42','timbrec43','timbrec44','timbrec45','timbrec46','timbrec47','timbrec48',
                                      'timbrec49','timbrec50','timbrec51','timbrec52','timbrec53','timbrec54','timbrec55','timbrec56','timbrec57','timbrec58','timbrec59','timbrec60',
                                      'timbrec61','timbrec62','timbrec63','timbrec64','timbrec65','timbrec66','timbrec67','timbrec68','timbrec69','timbrec70','timbrec71','timbrec72',
                                      'timbrec73','timbrec74','timbrec75','timbrec76','timbrec77','timbrec78'])
X_predict = scaler.transform(df2)
X_predict_sgd = scaler.transform(df2)

y_predict = model.predict(X_predict)
y_predict_sgd = model_sgd.predict(X_predict_sgd)
