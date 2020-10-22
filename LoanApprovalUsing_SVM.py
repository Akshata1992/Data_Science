# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:54:44 2020

@author: AKSHATA
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:25:06 2020

@author: AKSHATA
"""

#import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,f1_score
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
train_data = data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
test_data = data['Loan_Status']

x_train,x_test,y_train,y_test = train_test_split(train_data,test_data,test_size=0.25,random_state=0)
print("Shape of train data: ")
print(x_train.shape,y_train.shape)
print()
print("Shape of test data: ")
print(x_test.shape,y_test.shape)

#Normalize the data
norm = MinMaxScaler().fit(x_train)

# transform training data
X_train_norm = norm.transform(x_train)

# transform testing dataabs
X_test_norm = norm.transform(x_test)

#define the model and train or fit the model
#model = LogisticRegression()
#model = DecisionTreeClassifier(max_depth=3,min_samples_leaf = 55)
model = svm.LinearSVC(random_state=20)
model.fit(X_train_norm,y_train)

#predict the model
preds = model.predict(X_test_norm)

#print the accuracy of the model
print()
print("Accuracy of our model...")
print(accuracy_score(y_test, preds))
print()
print("Test F1 Score: ",f1_score(y_test,preds))

#build confusion metrics
print()
print("Confusion Matrix on Test Data")
cm = pd.crosstab(y_test, preds, rownames=['True'], colnames=['Predicted'], margins=True)