# -*- coding: utf-8 -*-
"""
Logistic linear regression 1
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load data
# corr between lag variables and today's returns are close to zero
# that means little corr between today's returns and previous day's returns
df = pd.read_csv('Smarket.csv', usecols=range(1,10))
print(df.corr())

#plot
fig = plt.figure(figsize=(10, 3)) # the 10 here is width, the 3 is height
ax = fig.add_subplot(111)
plt.plot(df.Volume, 'ro', color='black')
plt.xlabel("Days")
plt.ylabel("Volume(billions)")