#!/usr/bin/env python

def thetimestamp ():

    now = datetime.datetime.now()
    thetimestamp = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    return thetimestamp


from sklearn.datasets import load_boston
import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import plotly.plotly as py
#import plotly.tools as tls
import seaborn as sns

#boston = load_boston()
bst = load_boston()




# Create the Dataframe for reference
x = pd.DataFrame(bst.data)
x.columns = bst.feature_names
y=pd.DataFrame(bst.target)
y.columns=['MEDV']
boston = pd.concat([x, y], axis=1)


############################################################################################
### Linear Regression
############################################################################################
#Predicting Home Prices: a Simple Linear Regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bst.data, bst.target)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
expected = y_test

plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.savefig('SCT_predicted_true_price_linear_'+ thetimestamp() + '.png')
plt.close()
print('\n...Predicted True Price Scatter Chart on Full Data (Linear Regression)\n')
print("RMS: %s" % np.sqrt(np.mean((predicted - expected) ** 2)))

############################################################################################
########### Gradient Boosting Regression
############################################################################################
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
expected = y_test
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.savefig('SCT_predicted_true_price_gradient_boosted_'+ thetimestamp() + '.png')
plt.close()
print('\n...Predicted True Price Scatter Chart on Full Data (Gradient Boosting Regressor)')
############################################################################################
### Histogram - Price Target Data
############################################################################################
plt.figure(figsize=(4, 3))
plt.hist(bst.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()
#plt.show()
plt.savefig('HST_' + thetimestamp() + '.png')
print('\n...Price Target Data Histogram Generated')
plt.close()
############################################################################################
### Scatter Charts Data vs Target
############################################################################################
for index, feature_name in enumerate(bst.feature_names):
    #plt.figure(figsize=(4, 3))
    plt.figure(figsize=(8, 5))
    plt.scatter(bst.data[:, index], bst.target)
    #plt.ylabel('Price', size=15)
    plt.ylabel('MEDV', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
    plt.savefig('SCT_' + feature_name + '_'+ thetimestamp() + '.png')
    plt.close()

print('\n...Scatter Charts Feature vs Target Generated')
#print("RMS: %r " % np.sqrt(np.mean((predicted - expected) ** 2)))
############################################################################################
###DecisionTreeRegressor
############################################################################################
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor().fit(bst.data, bst.target)
predicted = clf.predict(bst.data)
expected = bst.target
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.savefig('SCT_TreeRegressor'+ thetimestamp() + '.png')
plt.close()
print('\n...Decision Tree Regressor Scatter Chart Generated')
############################################################################################
### DISTPLOT - Not Presently Required
############################################################################################
#boston.isnull().sum()
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.distplot(boston['MEDV'], bins=30)
#plt.show()
############################################################################################
#HEATMAP - correlation matrix
############################################################################################
sns.set(rc={'figure.figsize':(11.7,8.27)})
correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()
plt.savefig('Heatmap_' + thetimestamp() + '.png')
print('\n...Heatmap Generated\n')
############################################################################################
### Preparing Data for Training Model
############################################################################################
#R2 coefficient of determination is a measure of how well regression predictions approximate
#the real data points. An R2 of 1 indicates that the regression predictions perfectly fit the data.
#Root Mean Square Error (RMSE) measures how much error there is between two data sets. It compares
#a predicted value and an observed or known value.The function that has the smaller
#RMSE is the better estimator.
#X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
from sklearn.metrics import r2_score
X = pd.DataFrame(np.c_[boston['CRIM'], boston['PTRATIO']], columns = ['CRIM','PTRATIO'])
Y = boston['MEDV']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print('...Preparing the data for training the model\n')
print('X_train: ' +  str(X_train.shape))
print('X_test: ' + str(X_test.shape))
print('Y_train: ' + str(Y_train.shape))
print('Y_test: ' + str(Y_test.shape))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("\nModel performance testing and training performed on CRIM, PTRATIO vs MDEV\n")

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

###############################################################################
###############################################################################
#Gradient Boosting regression
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
X, y = shuffle(bst.data, bst.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
# #############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print('\nGradientBoostingRegressor Model MSE')
print("Mean Squared Error: %.4f" % mse)
############################################################################################
### Create Relative Importance and Training-Test Set Deviance
############################################################################################
# Plot training deviance
# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, bst.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
#plt.show()
plt.savefig('GBR_Relative_Importance_'+ thetimestamp() + '.png')
print('\n...Relative Importance and Training-Test Set Deviance Chart Generated\n')
###############################################################################
###############################################################################








print('\n\n\n -> Execution Completed')
