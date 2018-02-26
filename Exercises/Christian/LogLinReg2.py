# -*- coding: utf-8 -*-
"""
Logistic linear regression 2
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, classification_report
import sys

# load data
# corr between lag variables and today's returns are close to zero
# that means little corr between today's returns and previous day's returns
df = pd.read_csv('Smarket.csv', usecols=range(1,10))
print(df.corr())

#fit log model
formula = 'Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume'
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

#print out coeffieients, pvalues, dep. variables
print("Coeffieients: ", result.params, "\n")
print("p-Values: ", result.pvalues, "\n")
print("Dependent variables: ", result.model.endog_names, "\n")

#make a prediction if the market will go up or down
predictions = result.predict()
print("Predictions: ", predictions[0:10], "\n")

#make a prediction matrix based on the value of the predictions
#if x<0.5 then up, else down x
#the model predicted 507 days up, 145 days down.
predictions_matrix = ["Up" if x < 0.5 else "Down" for x in predictions]
print(confusion_matrix(df["Direction"], predictions_matrix), "\n")
print(classification_report(df["Direction"], predictions_matrix, digits=3), "\n")
#error rate = 1-0.522 = 47%
corr_predictions = (507+145)/1250
print("Percent correct predictions: ", corr_predictions)

#pull train data from year 2001 to 2004
#pull test data from only 2005
x_train = df[:sum(df.Year<2005)][:]
y_train = df[:sum(df.Year<2005)]['Direction']

x_test = df[sum(df.Year<2005):][:]
y_test = df[sum(df.Year<2005):]['Direction']

#fit a logistic reg model using observations before 2005
model = smf.glm(formula=formula, data=x_train, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

#compute predictions for 2005 and compare with the actual movements
predictions = result.predict(x_test)
predictions_matrix = ["Up" if x < 0.5 else "Down" for x in predictions]
print(classification_report(y_test, predictions_matrix, digits=3), "\n")
#error rate = 1-0.480 = 52%
#hard to predict future market performance from previous day's returns

#refit a log reg model using just lag1 and lag2 because they have low pvalues
formula = 'Direction ~ Lag1+Lag2'
model = smf.glm(formula=formula, data=x_train, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

#compute predictions for 2005 and compare with the actual movements
predictions = result.predict(x_test)
predictions_matrix = ["Up" if x < 0.5 else "Down" for x in predictions]
print(confusion_matrix(x_test["Direction"], predictions_matrix), "\n")
print(classification_report(y_test, predictions_matrix, digits=3), "\n")
corr_predictions = (35+106)/252
print("Percent correct predictions: ", corr_predictions)
#error rate = 1-0.560=0.44%
#we get a better error rate by removing predictors


















