# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:34:43 2021

@author: vikas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('AirPassengers.csv')

df.columns

r1 = np.arange(1, len(df)+1)
r1

df

plt.plot(r1, df['#Passengers'])


p: the number of lag observations in the model; also known as the lag order.
d: the number of times that the raw observations are differenced; also known as the degree of differencing.
q: the size of the moving average window; also known as the order of the moving average.



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(r1.reshape((-1,1)), df['#Passengers'])
y_pred = model.predict(r1.reshape((-1,1)))

y_pred

plt.scatter(r1.reshape((-1,1)), df['#Passengers'])
plt.scatter(r1.reshape((-1,1)), y_pred)





from statsmodels.tsa.arima_model import ARIMA

# Read the AirPassengers dataset 
airline = pd.read_csv('AirPassengers.csv', 
					index_col ='Month', 
					parse_dates = True) 

# Print the first five rows of the dataset 
airline.head() 

type(airline)

r = ARIMA(airline['#Passengers'],(3,0,3))

r = r.fit()

import datetime as dt

pred = r.predict(start = dt.datetime(1949,1,1), end=dt.datetime(1960,12,1))

y_pred = r.predict(start = dt.datetime(1961,1,1), end=dt.datetime(1966,12,1))


dt_dat = pd.date_range(dt.datetime(1961,1,1),dt.datetime(1966,12,1), freq='M')
len(dt_dat)
len(y_pred)

plt.scatter(airline.index, airline['#Passengers'])
#plt.scatter(airline.index, pred)
plt.scatter(dt_dat, y_pred[:-1])



'''
pred = r.predict(start = len(airline), end = (len(airline)-1) + 3 * 12)


airline['#Passengers'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 
pred.plot(legend = True)
'''



