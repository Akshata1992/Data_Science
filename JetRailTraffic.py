# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:23:15 2020

@author: AKSHATA
"""

#import all neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

#end of dec 2013 
df = pd.read_csv("Train.csv", nrows = 11856)
#end of october 2013
train = df[:10392]
#remaining 2 months for testing
test = df[10392:]

#aggregating dataset based on daily
df.Timestamp = pd.to_datetime(df.Datetime,format = '%d-%m-%Y %H:%M')
df.index = df.Timestamp
df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format = '%d-%m-%Y %H:%M')
train.index = train.Timestamp
train = train.resample('D').mean()
test.Timestamp = pd.to_datetime(test.Datetime,format = '%d-%m-%Y %H:%M')
test.index = test.Timestamp
test = test.resample('D').mean()
print(train.head())

#visualize with the plot
train.Count.plot(figsize = (15,8), title = 'Daily Ridership', fontsize=14)
test.Count.plot(figsize = (15,8), title = 'Daily Ridership', fontsize=14)
plt.show()

#Using time series model Seasonal ARIMA
y_hat = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count,order=(2,1,4),seasonal_order=(0,1,1,7)).fit()
y_hat['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic = True)
plt.figure(figsize=(16,8))
plt.plot(train['Count'],label='Train')
plt.plot(test['Count'],label='Test')
plt.plot(y_hat['SARIMA'],label='SARIMA')
plt.legend(loc='best')
plt.show()

#compute RMSE
rms = sqrt(mean_squared_error(test.Count, y_hat.SARIMA))
print(rms)
