# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:19:30 2018

@author: cml
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsClassifier
from patsy import dmatrices

#get the data
df = pd.read_csv('Smarket.csv', usecols=range(1,10))
#print(df.corr())

#get training data before 2005 and test data for 2005
formula = 'Direction~Lag1+Lag2'
y_train, x_train = dmatrices(formula, df.query('Year < 2005'), return_type='dataframe')
y_test, x_test = dmatrices(formula, df.query('Year >= 2005'), return_type='dataframe')

#chop the data up to remove unnecessary colums.
x_train_chop = x_train.iloc[:,1:3] # argument 1
y_train_chop = y_train.iloc[:,1:2] #argument 3
x_test_chop = x_test.iloc[:,1:3] # argument 2
y_test_chop = y_test.iloc[:,1]

#make KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_chop, y_train_chop) 

print('Classes: ', knn.classes_, "\n")

#make predictions for 2005 using knn and compare
#K = 1
x_test_labels = knn.predict(x_test_chop)
print('Accuracy (K=1): ', np.mean(y_test_chop==x_test_labels), "\n")
#50%

#make predictions using K = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_chop, y_train_chop) 
x_test_labels = knn.predict(x_test_chop)
print('Accuracy (K=3): ', np.mean(y_test_chop==x_test_labels), "\n")
#53%






