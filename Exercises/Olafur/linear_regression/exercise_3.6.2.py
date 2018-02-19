
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as formula

import matplotlib.pyplot as plot


data = pd.read_csv("Boston.csv")
data = data.drop("ID", axis=1)

# Python
#X = data[["lstat"]]
#y = data[["medv"]]

#sm.add_constant(X)

#results = sm.OLS(y, X).fit()

# R formula

results = formula.ols("medv ~ lstat", data=data).fit()

print results.summary()

print "------------------------------------------------------------------------------\n"

print "Predict for 10: \n{}\n".format(results.predict({"lstat":[5,10,15]}))


print "------------------------------------------------------------------------------\n"

sm.graphics.plot_partregress("medv", "lstat", [], data=data, obs_labels=False).show()


raw_input()
