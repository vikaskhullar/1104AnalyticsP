# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 14:53:48 2021

@author: vikas
"""

import numpy as np
from nsepy import get_history
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


pd.options.display.max_rows=50
pd.options.display.max_columns=10

start_date = datetime.datetime.today().date() - datetime.timedelta(days=366)

start_date

end_date = datetime.datetime.today().date()

panel_data = get_history(symbol='SBIN',
                   start=start_date,
                   end=end_date)

panel_data.columns
panel_data.to_csv('SBIN.csv')

close = panel_data['Close']

close.plot()

close[10:100]
all = pd.date_range(start=start_date, end=end_date)
all


close = close.reindex(all)
close
close.plot()

close.fillna(0).plot()


close = close.fillna(method='ffill')
close
close.plot()

#close = close.fillna(method='bfill')
#close


close_df= pd.DataFrame(close)
close_df.columns =['price'] 
close_df.columns
close_df.plot()


from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(close_df['price'], model ='multiplicative') 

# ETS plot 
result.plot() 



import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(close_df, order=(1, 2, 2), seasonal_order=(1, 2, 2,12))
results = mod.fit()
results


import itertools

p = d = q = range(0, 3)

pdq = list(itertools.product(p, d, q))
print(pdq)


seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

print('Few parameter combinations are:',seasonal_pdq )


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(close_df, order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = model.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
        
        
pred  = results.predict(0, len(close_df))

close_df[100:150].plot()
pred[100:150].plot()



from datetime import timedelta


#pred  = results.get_prediction(end_date, end_date + timedelta(days=pre), dynamic=False)
pred  = results.get_prediction(len(close_df)+1, len(close_df)+100, dynamic=False)
pred


pred_ci = pred.conf_int()

pred_ci

prediction=pred.predicted_mean
prediction

prediction.plot()
close.plot()















Ypred  = results.get_prediction(start = 0, end=10 , dynamic=False)


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
    
    
    
    
    





