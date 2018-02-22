# -*- coding: utf-8 -*-
"""
Multiple linear regression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm

data = datasets.load_boston()

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df[["LSTAT","AGE"]] # our independent variable
y = target["MEDV"] # our dependent variable
X = sm.add_constant(X) # add an intercept (beta_0) to our model

model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

print(model.summary())
print('predictions: ', predictions[0:5])
print('residuals: ', model.resid)

#make a linear model
lm = linear_model.LinearRegression()
#X = X.reshape(-1, 1) # when not using a constant beta_0
lm.fit(X,y)

print('lm score: ', lm.score(X,y))
print('Coefficients: \n', lm.coef_)

# Plot
fig = plt.figure(figsize=(10, 3)) # the 10 here is width, the 3 is height
ax = fig.add_subplot(111)
plt.scatter(y, predictions, color='black')
plt.plot(y, y, color='blue', linewidth=3)
plt.xlabel("Input")
# plt.ylabel("Response")

# # Use only one feature
# boston_X = boston.data[:, np.newaxis, 2]

# # Split the data into training and testing sets
# boston_X_train = boston_X
# boston_X_test = boston_X

# # Split the targets into training/testing sets
# boston_y_train = boston.target
# boston_y_test = boston.target

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model
# regr.fit(boston_X_train, boston_X_train)

# # Make predictions using the testing set
# boston_y_pred = regr.predict(boston_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(boston_y_test, boston_y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(boston_y_test, boston_y_pred))

# # Plot outputs
# plt.scatter(boston_X_test, boston_y_test,  color='black')
# plt.plot(boston_X_test, boston_y_pred, color='blue', linewidth=3)

# plt.xlabel("Input")
# plt.ylabel("Response")

# plt.xticks(())
# plt.yticks(())

# plt.show()