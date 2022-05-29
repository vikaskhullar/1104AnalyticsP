#Topic ---- TS - ARIMA - simple case
#%%%

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


'''
An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that 
uses time series data to either better understand the data set or to predict future trends. 

Understanding Autoregressive Integrated Moving Average (ARIMA)
An autoregressive integrated moving average model is a form of regression analysis that 
gauges the strength of one dependent variable relative to other changing variables. 
The model's goal is to predict future securities or financial market moves by examining 
the differences between values in the series instead of through actual values.


An ARIMA model can be understood by outlining each of its components as follows:

Autoregression (AR) refers to a model that shows a changing variable that regresses on its 
own lagged, or prior, values.

Integrated (I) represents the differencing of raw observations to allow for the time series 
to become stationary, i.e., data values are replaced by the difference between the data values 
and the previous values.

Moving average (MA) incorporates the dependency between an observation and a residual error 
from a moving average model applied to lagged observations.

Each component functions as a parameter with a standard notation. For ARIMA models, a standard 
notation would be ARIMA with p, d, and q, where integer values substitute for the parameters to 
indicate the type of ARIMA model used. The parameters can be defined as:

p: the number of lag observations in the model; also known as the lag order.
d: the number of times that the raw observations are differenced; also known as the degree of differencing.
q: the size of the moving average window; also known as the order of the moving average.

In a linear regression model, for example, the number and type of terms are included. A 0 value, 
which can be used as a parameter, would mean that particular component should not be used in the model. 
This way, the ARIMA model can be constructed to perform the function of an ARMA model, or even simple 
AR, I, or MA models.

Autoregressive Integrated Moving Average and Stationarity
In an autoregressive integrated moving average model, the data are differenced in order to make 
it stationary. A model that shows stationarity is one that shows there is constancy to the data 
over time. Most economic and market data show trends, so the purpose of differencing is to remove any 
trends or seasonal structures. 

Seasonality, or when data show regular and predictable patterns that repeat over a calendar year, 
could negatively affect the regression model. If a trend appears and stationarity is not evident, 
many of the computations throughout the process cannot be made with great efficacy.


ARIMA models are generally denoted ARIMA(p,d,q) where parameters p, d, and q 
are non-negative integers, p is the order (number of time lags) of the 
autoregressive model, d is the degree of differencing (the number of times 
the data have had past values subtracted), and q is the order of the 
moving-average model. 
Auto-Regressive Integrated Moving-Average (ARIMA)

"AR", "I" or "MA" from the acronym describing the model. 
For example, ARIMA (1,0,0) is AR, ARIMA(0,1,0) is I, and ARIMA(0,0,1) is MA


An ideal time series has stationarity. 
That means that a shift in time doesn’t cause a change in the shape of the distribution.
 Unit root processes are one cause for non-stationarity.
If you have unit roots in your time series, a series of successive differences, d, can transform the time series into one with stationarity. The differences are denoted by I(d), where d is the order of integration. Non-stationary time series that can be transformed in this way are called series integrated of order k. Usually, the order of integration is either I(0) or I(1); It’s rare to see values for d that are 2 or more.

'''



dates = pd.date_range('2012-07-09','2012-07-30')

dates
series = [43.,32.,63., 98.,65.,78.,23., 35.,78.,56.,45., 45.,56.,6.,63.,45., 64.,34.,76.,34., 14., 54. ]


res = pd.Series(series, index=dates)
res

plt.plot(res)


r = ARIMA(res,(3,0,3))

r = r.fit()

pred = r.predict(start='2012-07-10', end='2012-07-30')

plt.plot(dates,res)
plt.plot(dates[1:], pred)


ypred = r.predict(start='2012-07-30', end='2012-08-31')

date_pred =pd.date_range('2012-07-30','2012-08-31')


plt.plot(dates,res)
plt.plot(dates[1:], pred)
plt.plot(date_pred, ypred)

#https://stackoverflow.com/questions/36717603/predict-statsmodel-argument-error
#%%%

