# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:58:07 2020

@author: AKSHATA
"""

#import all necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt


#read dataset using pandas csv
train_data = pd.read_csv('train_v9rqX0R.csv')
test_data = pd.read_csv('test_AbJTz2l.csv')

#dealing with missing values
train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(),inplace=True)
test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(),inplace=True)

train_data['Outlet_Size'].fillna('Medium',inplace=True)
test_data['Outlet_Size'].fillna('Medium',inplace=True)

train_data = pd.get_dummies(train_data)
test_data  = pd.get_dummies(test_data)


train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']

print(train_x.shape,train_y.shape)

#define the model
model = LinearRegression()

#train the model
model.fit(train_x,train_y)

print("Model coeff: ", model.coef_)
print("Model intercept: ", model.intercept_)

#predict the model
preds = model.predict(train_x)

#model evaluate
model_eval = sqrt(mean_squared_error(train_y, preds))
print("MSE of model..")
print(model_eval)


