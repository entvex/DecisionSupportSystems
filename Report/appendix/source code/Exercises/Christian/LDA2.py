# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:51:16 2018

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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from patsy import dmatrices

#get the data
df = pd.read_csv('Smarket.csv', usecols=range(1,10))
#print(df.corr())

#get training data before 2005 and test data for 2005
formula = 'Direction~Lag1+Lag2'
y_train, x_train = dmatrices(formula, df.query('Year < 2005'), return_type='dataframe')
y_test, x_test = dmatrices(formula, df.query('Year >= 2005'), return_type='dataframe')

#chop the data up to remove unnecessary colums.
x_train_chop = x_train.iloc[:,1:3]
y_train_chop = y_train.iloc[:,1:2]
x_test_chop = x_test.iloc[:,1:3]
y_test_chop = y_test.iloc[:,1]

#perform QDA fit
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train_chop, y_train_chop)

print('Classes: ', qda.classes_, "\n")
print('Priors: ', qda.priors_, "\n")
print('Group means: ', qda.means_, "\n")

#make the predictions for 2005
#compare the actual data from 2005 with the predicated data
x_test_labels = qda.predict(x_test_chop)
print('Accuracy: ', np.mean(y_test_chop==x_test_labels), "\n")
#60%

