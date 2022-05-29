# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:04:32 2022

@author: vikas
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('AirPassengers.csv')

df=df.fillna(method='ffill')
df.plot()


import numpy as np
from statsmodels.tsa.arima_model import ARIMA

#p d q


len(df)

dt = pd.date_range('1949-01-01','1961-01-01', freq='m')

len(dt)

df.columns

r = ARIMA(df['#Passengers'], (3,0,3))

r = r.fit()

pred = r.predict(1, 144)

ypred = r.predict(1, 190)
len(pred)

df.columns
plt.plot(range(0,144), df['#Passengers'])
plt.plot(range(0,144), pred)
plt.plot(range(0,190), ypred)




import numpy as np

l1 = np.array(range(1,20))

l3=[]
for i in l1:
    l3.append(i**2)

l3

xt = np.linspace(0,2500, 26)

plt.plot(range(0, len(l2)), l2)
plt.yticks(xt)


import numpy as np

l1 = np.array(range(1,50))

l2=[]
for i in l1:
    l2.append(i**2)

l2

plt.plot(range(0, len(l3)), l3)
plt.plot(range(0, len(l2)), l2)
plt.yticks(xt)



'''
r = ARIMA(l1, (3,1,0))
r = r.fit()
pred = r.predict(1, 144)
ypred = r.predict(1, 170)
len(pred)

plt.plot(range(0,144), df['#Passengers'])
plt.plot(range(0,144), pred)
plt.plot(range(0,170), ypred)
'''


from statsmodels.tsa.seasonal import seasonal_decompose 


airline = pd.read_csv('AirPassengers.csv', 
					index_col ='Month', 
					parse_dates = True) 

airline = airline.fillna(method='ffill')
# Print the first five rows of the dataset 
airline.head() 

# ETS Decomposition 
result = seasonal_decompose(airline['#Passengers'], model ='multiplicative') 

result.plot() 



from statsmodels.tsa.statespace.sarimax import SARIMAX 

# Train the model on the full dataset 
model = SARIMAX(airline['#Passengers'], 
						order = (3, 1, 3), 
						seasonal_order =(3, 1, 3, 12)) # 4 for Quartely, 12 for Monthly, 52 for Weakly
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(airline), 
						end = (len(airline)) + 10 * 12, 
						typ = 'levels').rename('Forecast') 

forecast

plt.plot(range(0,len(airline['#Passengers'])), airline['#Passengers'])
plt.plot(range(144,144+len(forecast)), forecast)


# Case Study

from pandas_datareader import data

import datetime

start_date = datetime.datetime.today().date() - datetime.timedelta(days=91)

start_date

end_date = datetime.datetime.today().date()

panel_data = data.DataReader('AWL.NS', 'yahoo', start_date, end_date)

panel_data.columns

close = panel_data['Close']
close
close.plot()


all = pd.date_range(start=start_date, end=end_date)
all


close = close.reindex(all)

close.plot()



close = close.fillna(method='ffill')

close.plot()

close_df= pd.DataFrame(close)
close_df
close_df.columns =['price'] 
close_df.columns
close_df.plot()



from statsmodels.tsa.seasonal import seasonal_decompose 
result = seasonal_decompose(close_df['price'], model ='multiplicative') 
# ETS plot 
result.plot() 

list(result.seasonal)


import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(close_df, order=(3,1,3), seasonal_order=(3,1,3,12))
res = mod.fit()
pred = res.predict(start=len(close_df), end = len(close_df)+20)
pred

plt.plot(range(1, len(close_df)+1), close_df)
plt.plot(range(len(close_df), len(close_df)+len(pred)), pred)




panel_data.columns


close = panel_data['Volume']
close
close.plot()


all = pd.date_range(start=start_date, end=end_date)
all


close = close.reindex(all)

close.plot()



close = close.fillna(method='ffill')

close.plot()

close_df= pd.DataFrame(close)
close_df
close_df.columns =['price'] 
close_df.columns
close_df.plot()



from statsmodels.tsa.seasonal import seasonal_decompose 
result = seasonal_decompose(close_df['price'], model ='multiplicative') 
# ETS plot 
result.plot() 

list(result.seasonal)


import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(close_df, order=(3,1,5), seasonal_order=(3,1,3,4))
res = mod.fit()
pred = res.predict(start=len(close_df), end = len(close_df)+20)
pred

plt.plot(range(1, len(close_df)+1), close_df)
plt.plot(range(len(close_df), len(close_df)+len(pred)), pred)

























