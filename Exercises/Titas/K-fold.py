# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:01:34 2018

@author: titas
"""
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics

Data= pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/Auto.csv")
X = Data["horsepower"].values.reshape(-1,1) #
y = Data["mpg"].values.reshape(-1,1)
kf= KFold()
kf.n_splits = 10

ytests = []
ypreds = []

for train_index, test_index in kf.split(Data):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = linear_model.LinearRegression()
    model.fit(X = X_train, y = y_train)
    y_pred = model.predict(X_test)  
    ypreds += list(y_pred)
    ytests += list(y_test)
    ms_error = metrics.mean_squared_error(ytests, ypreds)
    print(ms_error)




