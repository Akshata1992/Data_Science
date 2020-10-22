# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:05:07 2020

@author: AKSHATA
"""

#import all necessary libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import metrics

#read the dataset
data = pd.read_csv('Iris.csv')
#print(data.columns)

#data visualization using seaborn
sb.pairplot(data,hue='Species')
#plt.show()

#label encode the species column
encoder = LabelEncoder()
data.Species = encoder.fit_transform(data.Species)
#print(data.head())

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
train_data = data[features]
#print(train_data.head())

test_data = data['Species']
#print(test_data.head())

x_train,x_test,y_train,y_test = train_test_split(train_data,test_data,test_size=0.25,random_state=0)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

#Initiatelize the model
model = LogisticRegression()

#fit or train the model
model.fit(x_train,y_train)

#print the predictions
preds= model.predict(x_test)
#print("Predicted value on test data:", encoder.inverse_transform(preds))

#check the accuracy of the model
print("Accuracy of our model...")
print(accuracy_score(y_test, preds))

#visualization
cm = metrics.confusion_matrix(y_test, preds)
print(cm)