# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:20:02 2020

@author: AKSHATA
"""

#import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#read the training dataset
data = pd.read_csv('train_ctrUa4K.csv')
#print(data.head())

#replace special characters in depedencies column
dict = {'3+':3}
data.replace(dict,regex=False,inplace=True)

#check for missing values
missing_count = data[data.columns[data.isnull().any()]].isnull().sum()

#dealing with categorical data first
#impute missing values for categorical features
data.fillna(data.select_dtypes(include='object').mode().iloc[0], inplace=True)
missing_count_updated = data[data.columns[data.isnull().any()]].isnull().sum()

#define categorical columns
cat_col = ['Gender','Married','Self_Employed','Education','Property_Area','Loan_Status']

#define label encoder for catgorical features
encoder = LabelEncoder()
for col in cat_col:
    data[col]=encoder.fit_transform(data[col].astype(str))
#print(data.head())
    
#dealing with numerical data
data.fillna(data.select_dtypes(include='number').mode().iloc[0],inplace=True)
missing_count_updated1 = data[data.columns[data.isnull().any()]].isnull().sum()

#dealing with outliers
#sns.boxplot(y=data['CoapplicantIncome'])

#Find the correlation between variables(for reference)
#corr_data = data.corr(method='pearson')

#scatter plot(for reference)
#sns.scatterplot(y=data['CoapplicantIncome'],x=data["LoanAmount"])

#define train data and test data and split the train and test data
train_data = data.drop(columns=['Loan_ID','Loan_Status','Credit_History'],axis=1)
test_data = data['Loan_Status']

x_train,x_test,y_train,y_test = train_test_split(train_data,test_data,test_size=0.25,random_state=0)

#Scale the value of X_train and x_test
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)
N,D = x_train.shape
print("Shape of train data: ")
print(x_train.shape,y_train.shape)
print()
print("Shape of test data: ")
print(x_test.shape,y_test.shape)

#Build the model in tensorflow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(D,), activation='sigmoid'))

#Define loss and optimizer
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fit the model
ret=model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=200)

#Evaluate the model - model.evaluate() - prints loss and accuracy
print("Train score= ", model.evaluate(x_train,y_train))
print("Test score = ", model.evaluate(x_test, y_test))