import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as formula
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv("Smarket.csv")

# Get subsets for later manipulation
train = data.loc[data["Year"] < 2005]
test = data.loc[data["Year"] == 2005]

print "\n------------------------------------------------\n"

r_formula = "Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume"
model = formula.glm(r_formula, data=data, family=sm.families.Binomial())
results = model.fit()

print results.summary()

print "\n------------------------------------------------\n"

print "Coefficients of the fitted model"
print results.params

print "\n------------------------------------------------\n"

print "Predicted values\n"

predictions = results.predict()
predictions_nominal = ["Up" if x < 0.5 else "Down" for x in predictions]

print "Confusion matrix: Full set"
print confusion_matrix(data["Direction"], predictions_nominal)
print "Classification report"
print classification_report(data["Direction"], predictions_nominal)

print "\n------------------------------------------------\n"

print "Create training dataset"

train_result = formula.glm(r_formula, data=train, family=sm.families.Binomial()).fit()

train_predictions = train_result.predict(test)
train_predictions_nominal = ["Up" if x < 0.5 else "Down" for x in train_predictions]

print "Confusion matrix: training set"
print confusion_matrix(test["Direction"], train_predictions_nominal)
print "Classification report"
print classification_report(test["Direction"], train_predictions_nominal)

print "\n------------------------------------------------\n"

print "Reduced formula\n"

reduced_model = formula.glm("Direction~Lag1+Lag2", data=train, family=sm.families.Binomial())
reduced_result = reduced_model.fit()

reduced_predictions = reduced_result.predict(test)
reduced_predictions_nominal = ["Up" if x < 0.5 else "Down" for x in reduced_predictions]

print "Confusion matrix: Reduced formula"
print confusion_matrix(test["Direction"], reduced_predictions_nominal)
print "Classification report"
print classification_report(test["Direction"], reduced_predictions_nominal)

raw_input()
