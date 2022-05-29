# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:33:32 2022

@author: vikas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([5,15,25,35,45,55])
y = np.array([5,20,14,32,22,38])  #1 dim

x
y

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

#model.fit(x,y)
#ValueError: Expected 2D array, got 1D array instead:

x.shape

x = x.reshape((-1,1))
x.shape

model.fit(x,y)

model.intercept_
model.coef_

r2 = model.score(x,y)
r2

y_pred = model.predict(x)
y_pred


plt.scatter(x,y)
plt.plot(x,y_pred)
plt.scatter(x,y_pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, y_pred)


#Case
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datahouse.csv')
df

df = df[df['area']<2400]
df
x = df['price'].values.reshape((-1,1))

x.shape

y = df['area'].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
r2 = model.score(x,y)

r2
y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.scatter(x,y_pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, y_pred)


plt.boxplot(y)

x
y



#Case
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Housing.csv')
df

df[df['price']>10000000]

len(df[df['price']>10000000])

df = df[df['price']<10000000]

df.describe()

df = df.dropna()

x = df['price'].values.reshape((-1,1))

x.shape

y = df['area'].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
r2 = model.score(x,y)

r2
y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.scatter(x,y_pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, y_pred)

plt.boxplot(y)



#Multidimention Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Real estate.csv')
df = df.drop(['n'], axis=1)
df = df.dropna()
df

df.columns
x = df.drop(['Y house price of unit area'], axis=1).values
x.shape

y = df['Y house price of unit area'].values
y.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
ypred = model.predict(x)

r2 = model.score(x,y)
r2

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, ypred)


'''
import seaborn as sns
sns.set_theme()
# Load the penguins dataset
penguins = sns.load_dataset("penguins")
penguins.columns

df.columns

# Plot sepal width as a function of sepal_length across days
g = sns.lmplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    height=5
)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Snoot length (mm)", "Snoot depth (mm)")

'''


pp = sns.pairplot(df, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig.subplots_adjust(top=0.93, wspace=0.3)


#Case

import seaborn as sns
from pydataset import data
df = data('mtcars')

pp = sns.pairplot(df, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig.subplots_adjust(top=0.93, wspace=0.3)



x = df['disp'].values.reshape((-1,1))
y = df['mpg'].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.scatter(x,y_pred)

r2 = model.score(x,y)
r2

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, ypred)



df.columns

x = df[['disp', 'hp', 'drat', 'wt' ]].values
y = df['mpg'].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)
y_pred = model.predict(x)


r2 = model.score(x,y)
r2

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, ypred)

sns.pairplot(df[['disp', 'hp', 'drat', 'wt', 'mpg']], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig.subplots_adjust(top=0.93, wspace=0.3)



df.columns
col = ['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear',
       'carb']

import seaborn as sns
sns.set_theme()
df.columns


for i in col:
    g = sns.lmplot(
        data=df,
        x=i, y="mpg",
        height=5)












































































































