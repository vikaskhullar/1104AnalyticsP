# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:01:24 2022

@author: vikas
"""


import math
X= list(range(-24,25))
X

y=[]

for x in X:
    y.append((1)/(1+math.exp(-x)))
y

import matplotlib.pyplot as plt
plt.scatter(X,y)




import numpy as np
x= np.array(list(range(-6,7))).reshape((-1,1))
x.shape

y = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
y.shape

x
y

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)
pred = model.predict(x)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.scatter(x,pred)
model.predict(np.array([-2]).reshape((-1,1)))



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x,y)

pred = model.predict(x)
plt.scatter(x,y, marker='*')
plt.scatter(x,pred, s=25)



#Case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('binary.csv')


x = df['gre'].values.reshape((-1,1))
y = df['admit'].values

x.shape
y.shape

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x,y)
pred = model.predict(x)
pred

plt.scatter(x,y)
plt.scatter(x,pred)


y = [0,0,1,1,1,1,0,1,1,1]
p = [1,1,1,1,0,1,1,0,1,1]


TP= 5
TN = 0
FP = 3
FN = 2
(TP +TN )/ (TP+TN+FP+FN)


from sklearn.metrics import accuracy_score
accuracy_score(y,p)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,p)

y = [0,0,1,1,1,1,1,1,1,1]
p = [1,1,1,1,1,1,1,1,1,1]

from sklearn.metrics import accuracy_score
accuracy_score(y,p)
y = [0,0,1,1,1,1,1,1,1,1]
p = [1,1,1,1,1,1,1,1,1,1]
print(classification_report(y, p))




#Case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('titanic_all_numeric.csv')

df.columns

x = df.drop(['survived'], axis=1).values

y = df['survived'].values

x.shape

y.shape


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x,y)
pred = model.predict(x)
pred


from sklearn.metrics import accuracy_score
accuracy_score(y,pred)

from sklearn.metrics import classification_report

print(classification_report(y, pred))


#Case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('smoker.csv')


df['TenYearCHD'].value_counts()
df = df.dropna()
df.columns

x = df.drop(['TenYearCHD'], axis=1).values

y = df['TenYearCHD'].values


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x,y)
pred = model.predict(x)
pred


from sklearn.metrics import accuracy_score
accuracy_score(y,pred)

from sklearn.metrics import classification_report

print(classification_report(y, pred))

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(x, y)
pd.Series(Ys).value_counts()



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(Xs,Ys)
pred = model.predict(Xs)
pred

from sklearn.metrics import classification_report
print(classification_report(Ys, pred))



# Case


from sklearn.model_selection import train_test_split
Xs.shape, Ys.shape

xtrain,xtest, ytrain,  ytest = train_test_split(Xs, Ys, test_size=0.3)
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(xtrain, ytrain)

pred = model.predict(xtest)
pred

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


#Case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('HRNum.csv')

df.columns

x = df.drop(['Unnamed: 0', 'Attrition'], axis=1).values
y = df[ 'Attrition'].values

df[ 'Attrition'].value_counts()


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(x, y)
pd.Series(Ys).value_counts()


from sklearn.model_selection import train_test_split
Xs.shape, Ys.shape

xtrain,xtest, ytrain,  ytest = train_test_split(Xs, Ys, test_size=0.3)
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

from sklearn.linear_model import LinearRegression

model.fit(xtrain, ytrain)

pred = model.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))




#without Smote

xtrain,xtest, ytrain,  ytest = train_test_split(x, y, test_size=0.3)
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

from sklearn.linear_model import LinearRegression

model.fit(xtrain, ytrain)

pred = model.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



#Decision Tree, Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('HRNum.csv')

df.columns

x = df.drop(['Unnamed: 0', 'Attrition'], axis=1).values
y = df[ 'Attrition'].values

df[ 'Attrition'].value_counts()


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(x, y)
pd.Series(Ys).value_counts()



from sklearn.model_selection import train_test_split
Xs.shape, Ys.shape

xtrain,xtest, ytrain,  ytest = train_test_split(Xs, Ys, test_size=0.3)
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(Xs, Ys)

pred = model.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


model.fit(Xs, Ys)

pred = model.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



#Case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df.columns

x= df.drop(['name'], axis=1).values
y = df['name'].values

from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain,  ytest = train_test_split(x, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(xtrain, ytrain)
pred = model.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


#Case
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df.columns

x= df.drop(['name'], axis=1).values
y = df['name'].values

from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain,  ytest = train_test_split(x, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(xtrain, ytrain)
pred = model.predict(xtest)

from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


#Case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data1.csv')

df.columns

x = df.drop(['purpose','not.fully.paid'], axis=1).values
y = df['not.fully.paid'].values



from imblearn.over_sampling import SMOTE
oversample = SMOTE()
Xs, Ys = oversample.fit_resample(x, y)
pd.Series(Ys).value_counts()

from sklearn.model_selection import train_test_split
Xs.shape, Ys.shape

xtrain,xtest, ytrain,  ytest = train_test_split(Xs, Ys, test_size=0.3)
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
pred
from sklearn.metrics import classification_report
print(classification_report(ytest, pred))


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
pred
from sklearn.metrics import classification_report
print(classification_report(ytest, pred))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
pred
from sklearn.metrics import classification_report
print(classification_report(ytest, pred))



#Regrssion Using Desicion tree and random forest


import seaborn as sns
from pydataset import data
df = data('mtcars')

x = df['disp'].values.reshape((-1,1))
y = df['mpg'].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
y_pred = model.predict(x)
r2 = model.score(x,y)
r2

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x,y)
y_pred = model.predict(x)
r2 = model.score(x,y)
r2


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x,y)
y_pred = model.predict(x)
r2 = model.score(x,y)
r2



































