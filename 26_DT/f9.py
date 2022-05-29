# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:19:39 2021

@author: vikas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics

df = pd.read_csv('titanic_all_numeric.csv')

df.dtypes

x = df.drop(['survived', 'age_was_missing'], axis=1).values
x.shape


y= df['survived'].values
y.shape


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

model = DecisionTreeClassifier()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_test
y_pred

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = model.predict_proba(x_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();





# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:19:39 2021

@author: vikas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics

df = pd.read_csv('loan_data.csv')

df.dtypes

x = df.drop(['not.fully.paid', 'purpose'], axis=1).values
x.shape


y= df['not.fully.paid'].values
y.shape


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_test
y_pred

y_test.unique

np.unique(y_test, return_counts=True)
np.unique(y_pred, return_counts=True)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = model.predict_proba(x_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together




plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();








