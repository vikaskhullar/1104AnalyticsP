# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:39:41 2020

@author: vikas
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime

pd.options.display.max_rows=50
pd.options.display.max_columns=10

datetime.datetime.today().date()

start_date = datetime.datetime.today().date() - datetime.timedelta(days=60)

start_date

end_date = datetime.datetime.today().date()

panel_data = data.DataReader('RELIANCE.NS', 'yahoo', start_date, end_date)

panel_data.head(10)
panel_data.columns
panel_data

close = panel_data['Close']

close
close.plot()

all = pd.date_range(start=start_date, end=end_date)
all

close = close.reindex(all)
close
close.plot()

close.fillna(0).plot()


close = close.fillna(method='ffill')
close = close.fillna(method='bfill')
close
close.plot()

#close = close.fillna(method='bfill')
#close

print(all_weekdays)
panel_data

close
close.head(10)

close.describe()

close_df= pd.DataFrame(close)
close_df
close_df.columns =['price'] 
close_df.columns
close_df.plot()


from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(close_df['price'], model ='multiplicative') 

# ETS plot 
result.plot() 

result.trend
result.seasonal

len(close_df)

import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(close_df, order=(3,1,3), seasonal_order=(3,1,3,12))
results = mod.fit()
results


from datetime import timedelta

pre =20
pred  = results.get_prediction(start = len(close_df), end=len(close_df)+pre, dynamic=False)

#pred  = results.get_prediction(start = end_date, end=end_date + timedelta(days=pre), dynamic=False)

pred

pred_ci = pred.conf_int()
pred_ci

prediction=pred.predicted_mean
prediction


plt.plot(close_df)
plt.plot(pred_ci)
plt.plot(prediction)
#plt.plot(date_pred, prediction)




Ypred  = results.get_prediction(start = start_date, end=end_date , dynamic=False)


Yprediction=Ypred.predicted_mean
Yprediction.shape


date_pred= pd.date_range (pd.to_datetime(end_date), end=pd.to_datetime(end_date) + timedelta(days=pre))

y_pred= pd.date_range (start_date, end_date)
y_pred.shape

plt.plot(close_df.index,close_df.price)
plt.plot(y_pred[1:],Yprediction[1:])

plt.plot(pred_ci)
plt.plot(date_pred, prediction)


from sklearn.metrics import r2_score, mean_absolute_error

r2_score(close_df[1:].price.values,Yprediction[1:].values)

mean_absolute_error(close_df[1:].price.values,Yprediction[1:].values)

AV= close_df[1:].price.values

FV = Yprediction[1:].values

mape = 0
for i in range(len(AV)):
    mape = mape + (abs(AV[i] - FV[i])/AV[i])

mape

mape = (mape/ len (AV)) *100
mape
mape
    
    
    
    
    





