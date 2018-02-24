import pandas as pd
data = pd.read_csv("https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/MASS/Boston.csv", index_col=0)
print(data.head())
print(data.shape)

# conventional way to import seaborn
#import seaborn as sns
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# visualize the relationship between the features and the response using scatterplots
#sns.pairplot(data, x_vars=['lstat','age'], y_vars='medv', size=7, aspect=0.7, kind='reg')

# create a Python list of feature names
feature_cols = ['lstat','age']

# use the list to select a subset of the original DataFrame
X = data[feature_cols].values

print(X)
print(X.shape)

# select a Series from the DataFrame
y = data['medv'].values

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X, y)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)

# make predictions on the testing set
y_pred = linreg.predict(X)

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(y, y_pred)))

# Plot outputs
import statsmodels.api as sm

X_1 = sm.add_constant(X)
result=sm.OLS(y,X_1).fit()
print( result.params )

#plt.scatter(X, y)
#plt.plot(X, y_pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()