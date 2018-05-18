# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:47:12 2018

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

#perform LDA fit
lda = LinearDiscriminantAnalysis()
lda.fit(x_train_chop, y_train_chop)

#Priors: 0.492 and 0.508.
#49.2% of the observations are days which the market went down
#50.8% of the observations are days which the market went up
#Group means: the averages used for lda
#Lag1+2 are negative on days where market increases,
#Lag1+2 are positive on days where market decreases.
#Coefs are used in LDA to predict market rise/decline
print('Priors: ', lda.priors_, "\n")
print('Group means: ', lda.means_, "\n")
print('Coefs: ', lda.coef_, "\n")

#make the predictions (classifications)
x_predict_labels = lda.predict(x_train_chop)
#get posterior probabilities for each class of X
x_predict_prob = lda.predict_proba(x_train_chop)

#testing step
#compare predicted model to the actual data
x_test_labels = lda.predict(x_test_chop)
x_test_prob = lda.predict_proba(x_test_chop)
print('Accuracy: ', np.mean(y_test_chop==x_test_labels), "\n")

#change the threshold to improve the accuracy. 2nd column
#of x_test_prob belongs to UP group. Default is 0.5
threshold = 0.5
print('Accuracy (0.5): ', np.mean(y_test_chop==(x_test_prob[:,1]>=threshold)), "\n")

threshold = 0.48
print('Accuracy (0.48): ', np.mean(y_test_chop==(x_test_prob[:,1]>=threshold)), "\n")

#predicted_sum = sum(y_train.iloc[:,1]==x_labels)
#predicted_mean = (predicted_sum/len(x_labels))
#mean is 0.51 which means 51% percent of the time we guess correctly

