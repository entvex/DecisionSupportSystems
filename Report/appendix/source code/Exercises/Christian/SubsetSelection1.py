#- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:16:21 2018

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

df = pd.read_csv('Hitters.csv').dropna()

#split data
X = df.iloc[:,1:19].drop('League', 1).drop('Division', 1) # our independent variable
y = df["Salary"] # our dependent variable

selector = SelectKBest(f_regression, k=15)
selector.fit(X, y)
print("Pvalues: ", selector.pvalues_)
    
#get the p values for each feature by scoring using RSS
#and remove all but the k highest scoring features
scores = -np.log10(selector.pvalues_)

predictors = ["AtBat", "Hits", "HmRun", "Runs", "RBI", "Walks",
             "Years", "CAtBat", "CHits", "CHmRun", "CRuns",
             "CRBI", "CWalks", "PutOuts", "Assits", "Errors"]

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#initial slicing
#train, test = np.split(df.sample(frac=1), [int(0.5*len(df))])

#split the data by formula
#formula = 'mpg~horsepower'
#y_train, x_train = dmatrices(formula, train, return_type='dataframe')
#y_test, x_test = dmatrices(formula, test, return_type='dataframe')