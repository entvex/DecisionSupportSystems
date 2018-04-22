# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:00:36 2018

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
from patsy import dmatrices
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso

df = pd.read_csv('Hitters.csv').dropna()

#remove unusable data
df.iloc[:,1:19].drop('League', 1).drop('Division', 1)

#split data
train, test = np.split(df.sample(frac=1), [int(0.5*len(df))])
formula = 'Salary~AtBat+Hits+HmRun+Runs+RBI+Walks+Years+CAtBat+CHits+CHmRun+CRuns+CRBI+CWalks+PutOuts+Assists+Errors'
y_train, x_train = dmatrices(formula, train, return_type='dataframe')
y_test, x_test = dmatrices(formula, test, return_type='dataframe')

#our two methods
ridge = Ridge()
lasso = Lasso(max_iter=10000)
alphas_selected = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for a in alphas_selected:
    ridge.set_params(alpha=a)
    lasso.set_params(alpha=a)
    ridge.fit(scale(x_train), scale(y_train))
    lasso.fit(scale(x_train), scale(y_train))
    preds_ridge = ridge.predict(x_train)
    preds_lasso = lasso.predict(x_train)
    print('MSE RIDGE alpha=', a, ":", mean_squared_error(y_test.iloc[0:131], preds_ridge))
    print('MSE LASSO alpha=', a, ":", mean_squared_error(y_test.iloc[0:131], preds_lasso))

#for all alphas, make a ridge fit, save the coefs and plot them with lambda
#courtesy of http://scikit-learn.org/stable/auto_examples/linear_model/
#plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py
alphas_all = 10**np.linspace(10,-2,100)*0.5

X = df.iloc[:,1:19].drop('League', 1).drop('Division', 1)
y = df["Salary"]
coefs_ridge = []
for a in alphas_all:
    ridge.set_params(alpha=a)
    ridge.fit(scale(X), y)
    coefs_ridge.append(ridge.coef_)
    
ax = plt.gca()
ax.plot(alphas_all, coefs_ridge)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Lamda')
plt.ylabel('weights')
plt.title('Ridge coefficients as the Lamda changes');
plt.show()

coefs_lasso = []
for a in alphas_all:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X), y)
    coefs_lasso.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas_all, coefs_lasso)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Lamda')
plt.ylabel('weights')
plt.title('Lasso coefficients as the Lamda changes');
plt.show()