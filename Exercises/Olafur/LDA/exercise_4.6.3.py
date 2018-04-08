#! /usr/bin/python

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as formula

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data  = pd.read_csv("Smarket.csv")
data["Direction"] = data["Direction"].astype("category", copy=True)

train = data.loc[data["Year"] <  2005]
test  = data.loc[data["Year"] == 2005]

prior_up   = float(len(train.loc[train["Direction"] == "Up"])) / float(len(train))
prior_down = float(len(train.loc[train["Direction"] == "Down"])) / float(len(train))

print "Prior probability UP: {}".format(prior_up)
print "Prior probability DOWN: {}".format(prior_down)

print "Mean Lag1 when UP: {}".format(train.loc[train["Direction"] == "Up"]["Lag1"].mean())
print "Mean Lag1 when DOWN: {}".format(train.loc[train["Direction"] == "Down"]["Lag1"].mean())
print "Mean Lag2 when UP: {}".format(train.loc[train["Direction"] == "Up"]["Lag2"].mean())
print "Mean Lag2 when DOWN: {}".format(train.loc[train["Direction"] == "Down"]["Lag2"].mean())

print train.loc[:, ["Lag1","Direction"]].corr()

#model = LinearDiscriminantAnalysis()
#model.fit(train.loc[:, ["Lag1","Lag2"]], train["Direction"].cat.codes)

#print model



