#!/usr/bin/env python

def thetimestamp ():

    now = datetime.datetime.now()
    thetimestamp = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    return thetimestamp


from argparse import ArgumentParser
from sklearn.datasets import load_boston

import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.plotly as py
import plotly.tools as tls

import seaborn as sns
#sns.set(style="darkgrid")


data = load_boston()


########### Linear Regression
#Predicting Home Prices: a Simple Linear Regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

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
plt.savefig('SCT_predicted_true_price_'+ thetimestamp() + '.png')
plt.close()

print("RMS: %s" % np.sqrt(np.mean((predicted - expected) ** 2)))




########### Gradient Boosting Regression
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

#x = pd.DataFrame(data.data)
#x.columns = data.feature_names
#y=pd.DataFrame(data.target)
#y.columns=['MEDV']
#data = pd.concat([x, y], axis=1)

#print(data.data.shape)
#print(data.target.shape)

plt.figure(figsize=(4, 3))
plt.hist(data.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()
#plt.show()
plt.savefig('HST_' + thetimestamp() + '.png')
plt.close()


for index, feature_name in enumerate(data.feature_names):
    #plt.figure(figsize=(4, 3))
    plt.figure(figsize=(8, 5))
    plt.scatter(data.data[:, index], data.target)
    #plt.ylabel('Price', size=15)
    plt.ylabel('MEDV', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
    plt.savefig('SCT_' + feature_name + '_'+ thetimestamp() + '.png')
    plt.close()

print("RMS: %r " % np.sqrt(np.mean((predicted - expected) ** 2)))


#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor().fit(data.data, data.target)
predicted = clf.predict(data.data)
expected = data.target
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.savefig('SCT_TreeRegressor'+ thetimestamp() + '.png')
plt.close()

######################################################



###############################################################################
###############################################################################
#Gradient Boosting regression
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
X, y = shuffle(data.data, data.target, random_state=13)
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
print("Mean Squared Error: %.4f" % mse)
# #############################################################################
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
plt.yticks(pos, data.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
#plt.show()
plt.savefig('GBR_Relative_Importance_'+ thetimestamp() + '.png')
###############################################################################
###############################################################################



#import statsmodels.api as sm


#print(data.data)
#print(data.feature_names)
#print(data.target)

#X = data['RM']
#X = data.data
#y = data.target

# Note the difference in argument order
#model = sm.OLS(y, X).fit()
#predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
#model.summary()


#sorted_dataframe = data.sort_values(c) # where c was the first column we were plotting
#sorted_dataframe = sorted_dataframe.reset_index(drop=True)
#for sorted_idx, y_pred in enumerate(data.feature_names[sorted_idx]):
#    print(data.feature_names[sorted_idx])




#X = data.data
#y = data.target

# Instantiate and train the classifier
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=1)
#clf.fit(X, y)
#KNeighborsClassifier(...)

# Check the results using metrics
#from sklearn import metrics
#y_pred = clf.predict(X)

#print(metrics.confusion_matrix(y_pred, y))



print('\n\n\n -> Completed')
