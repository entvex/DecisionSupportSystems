# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:43:48 2018

@author: cml
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.utils import resample

def alpha( X, Y ):
    return ((np.var(Y)-np.cov(X, Y))/(np.var(X)+np.var(Y)-2*np.cov(X ,Y)))

#randomly select 100 observations from range 1 to 1000 with replacement
def bootstrap (df):
    totalResult = 0
    for i in range(0,1000):
        dfsample = df.sample(frac=1, replace=True)
        X = dfsample.X[0:100]
        Y = dfsample.Y[0:100]
        result = alpha(X, Y)
        totalResult += result
    endResult = totalResult / 1000
    print("End result of bootstrap: ", endResult, "\n")

df = pd.read_csv('Portfolio.csv', usecols=range(1,3))
X = df.X[0:100]
Y = df.Y[0:100]
print('alpha: ', alpha(X,Y), "\n")

#run bootstrap
bootstrap(df)
#final output shows alpha = 0.58

#load auto data set to perform bootstrap to estimate
#the accuracy of a linear regression model
auto_df = pd.read_csv('Auto.csv', usecols=range(1,10))

#split data
auto_X = auto_df["horsepower"].values.reshape(-1,1) # our independent variable
auto_y = auto_df["mpg"].values.reshape(-1,1) # our dependent variable
auto_X = sm.add_constant(auto_X) #add constant to get intercept

#create linear model and calculate standard errors
#SE(B0)=0.717, SE(B1)=0.006
ols = sm.OLS(auto_y, auto_X).fit()
print(ols.summary())

#use bootstrapping
#it gives a more accurate estimate of the standard errors
#SE(B0)=0.448, SE(B1)=0.004
Xsamp, ysamp = resample(auto_X, auto_y, n_samples=1000)
ols_resample = sm.OLS(ysamp, Xsamp).fit()
print(ols_resample.summary())