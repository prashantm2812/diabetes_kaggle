# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 23:34:22 2019

@author: Prashant Maheshwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes.csv')
X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataset[['Outcome']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc1 = MinMaxScaler()
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 118, gamma = 0.02, random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [118, 117, 116, 115], 'kernel' : ['poly', 'sigmoid', 'rbf'], 'gamma' : [0.01, 0.012, 0.013, 0.014, 0.02], 'random_state' : [0, 11, 42]}]
parameters1 = [{'C' : [118, 119, 120, 121, 122], 'kernel' : ['linear'], 'gamma' : [0.01, 0.012, 0.013, 0.014, 0.02],
               'random_state' : [0, 11, 42]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 3,
                           verbose = 60, n_jobs = 1)
grid_search = grid_search.fit(X_train, y_train)
best_score1 = grid_search.best_score_
best_params = grid_search.best_params_