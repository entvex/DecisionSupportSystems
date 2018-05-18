
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as formula

import matplotlib.pyplot as plot


data = pd.read_csv("Boston.csv")
data = data.drop("ID", axis=1)

# R formula
# . operator is not defined. 
results = formula.ols("medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat", data=data).fit()

print results.summary()

raw_input()
