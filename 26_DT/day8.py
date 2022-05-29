# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 19:30:00 2021

@author: vikas
"""

from sklearn import tree
import numpy as np

X = np.array([[10], [20], [15], [30], [18], [17]])
Y = np.array([0,1,0,1,1,0])  #class labels -  0- play no, 1- play yes
X
Y
X.shape
Y.shape

clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)

tree.plot_tree(decision_tree= clf)








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')

df.dtypes
df.columns


df['not.fully.paid'].value_counts()



X = df.drop(['purpose','not.fully.paid'], axis=1).values
Y = df['not.fully.paid'].values



X.shape
Y.shape



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

pred = model.predict(X_test)

pred

from sklearn.metrics import classification_report

print(classification_report(Y_test, pred))






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')

df.dtypes
df.columns


df['not.fully.paid'].value_counts()



X = df.drop(['purpose','not.fully.paid'], axis=1).values
Y = df['not.fully.paid'].values


X.shape
Y.shape


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(X, Y)

pd.Series(Ys).value_counts()
Xs.shape
Ys.shape



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(Xs,Ys, test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


from sklearn.linear_model import LogisticRegression

model =LogisticRegression()

model.fit(X_train, Y_train)

pred = model.predict(X_test)

pred

from sklearn.metrics import classification_report

print(classification_report(Y_test, pred))









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')

df.dtypes
df.columns


df['not.fully.paid'].value_counts()



X = df.drop(['purpose','not.fully.paid'], axis=1).values
Y = df['not.fully.paid'].values


X.shape
Y.shape


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(X, Y)

pd.Series(Ys).value_counts()
Xs.shape
Ys.shape



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(Xs,Ys, test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

pred = model.predict(X_test)

pred

from sklearn.metrics import classification_report

print(classification_report(Y_test, pred))



import pickle
pickle.dump(model, open('DTModel.sav', 'wb'))






# Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df.columns

X = df.drop('name', axis=1).values
Y = df['name'].values

X.shape
Y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))


#Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))

from sklearn.neighbors import NearestCentroid
model = NearestCentroid()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred))



import pickle
pickle.dump(model, open('Model.sav', 'wb'))




res = classification_report(Y_test, pred)


pdf = pandas_classification_report(Y_test, pred)

type(pdf)

pdf.to_csv('pdf.csv')


from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support

def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    
    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total
    
    class_report_df['avg / total'] = avg
    return class_report_df.T









































