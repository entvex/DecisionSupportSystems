import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as formula

import matplotlib.pyplot as plot

data = pd.read_csv("Smarket.csv")
data = data.drop("ID", axis=1)


print "Describe dataset"

print data.describe()

print "\n------------------------------------------------\n"

print "Correlations"
print data.corr()

print "\n------------------------------------------------\n"

plot.plot(data["Volume"])

plot.show()

raw_input()

