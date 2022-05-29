# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 19:29:49 2021

@author: vikas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA


dates = pd.date_range('2012-07-09','2012-07-30')

dates

series = [43.,32.,63., 98.,65.,78.,23., 35.,78.,56.,45., 45.,56.,6.,63.,45., 64.,34.,76.,34., 14., 54. ]


res = pd.Series(series, index=dates)
res

plt.plot(res)

#p AR lag, d I 0, 1, q MA 3

r = ARIMA(res, (0,0,3))

r = r.fit()

pred = r.predict(start='2012-07-10', end='2012-07-30')

plt.plot(dates, res)
plt.plot(dates[1:], pred)


ypred = r.predict(start='2012-07-30', end='2012-08-31')

date_pred =pd.date_range('2012-07-30','2012-08-31')


plt.plot(dates,res)
plt.plot(dates[1:], pred)
plt.plot(date_pred, ypred)









