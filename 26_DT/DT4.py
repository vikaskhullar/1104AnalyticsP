# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:06:46 2021

@author: vikas
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv("iris.csv")

df.columns

#df = df.drop(['credit.policy', 'purpose'], axis=1)


y = df['name']

y


df = df.drop(['name'], axis =1)

X = df

df.dtypes

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
logreg = DecisionTreeClassifier()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred
y_test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cr=classification_report(y_test, y_pred)
print(cr)


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))


X_test = X_test.values

X_test[29]
y_test.values[29]

X_test = np.array([5.5 , 2.4, 1.8, 1.6])
logreg.predict(X_test.reshape(1,-1))

