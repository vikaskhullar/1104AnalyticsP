# Importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from statsmodels.tsa.seasonal import seasonal_decompose 

# Read the AirPassengers dataset 
airline = pd.read_csv('AirPassengers.csv', 
					index_col ='Month', 
					parse_dates = True) 

# Print the first five rows of the dataset 
airline.head() 

# ETS Decomposition 
result = seasonal_decompose(airline['#Passengers'], model ='multiplicative') 

result.plot() 



from statsmodels.tsa.statespace.sarimax import SARIMAX 

# Train the model on the full dataset 
model = SARIMAX(airline['#Passengers'], 
						order = (2, 1, 3), 
						seasonal_order =(2, 1, 3, 12)) # 4 for Quartely, 12 for Monthly, 52 for Weakly
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(airline), 
						end = (len(airline)) + 10 * 12, 
						typ = 'levels').rename('Forecast') 

forecast

# Plot the forecast values 

forecast1 = result.predict(start = 0, 
						end = 144, 
						typ = 'levels').rename('Forecast1')


airline['#Passengers'].plot(figsize = (12, 5), legend = True, lw=4, linestyle='dashed') 
forecast.plot(legend = True, lw=4) 
forecast1.plot(legend = True,lw=2) 




from statsmodels.tsa.statespace.sarimax import SARIMAX 

# Train the model on the full dataset 
model = SARIMAX(airline['#Passengers'], 
						order = (2, 1, 3), 
						seasonal_order =(2, 1, 3, 12)) # 4 for Quartely, 12 for Monthly, 52 for Weakly
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(airline), 
						end = (len(airline)) + 3 * 12, 
						typ = 'levels').rename('Forecast') 

forecast

# Plot the forecast values 

forecast1 = result.predict(start = 0, 
						end = 144, 
						typ = 'levels').rename('Forecast1')


airline['#Passengers'].plot(figsize = (12, 5), legend = True, lw=4, linestyle='dashed') 
forecast.plot(legend = True, lw=4) 
forecast1.plot(legend = True,lw=2) 


forecast2 = result.get_prediction(start = 0,end = 144)
a = forecast2.conf_int()['lower #Passengers']
a
b = forecast2.conf_int()['upper #Passengers']
b

plt.plot(range(0,len(forecast2.conf_int())), a)
plt.plot(range(0,len(forecast2.conf_int())), b)

forecast2.conf_int().plot()


forecast.conf_int()





from statsmodels.tsa.arima_model import ARIMA

airline
r = ARIMA(airline['#Passengers'],(3,0,3))

r = r.fit()


pred = r.predict(start = len(airline), end = (len(airline)-1) + 3 * 12)


airline['#Passengers'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 
pred.plot(legend = True)



pred = r.predict(start='2012-07-31', end='2012-08-31')
date_pred =pd.date_range('2012-07-31','2012-08-31')





