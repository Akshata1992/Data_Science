#import all necessary packages
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#read train ans test set
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())

#shape of the dataset
print("Shape of the dataset: ",train_data.shape)
print("Shape of the test dataset: ",test_data.shape)

#now predict the missing target variable in test dataset: Item_Outlet_sales
#Drop the targte variable from train data
train_x = train_data.drop(columns=['Item_Outlet_Sales'], axis = 1)
#separate the target variable to train_y
train_y = train_data['Item_Outlet_Sales']

#Similarlly obtain the target variable in test set
test_x = test_data.drop(columns = ["Item_Outlet_Sales"], axis =1)
test_y = test_data["Item_Outlet_Sales"]

#Define the model
model = LinearRegression()

#Fit the model
model.fit(train_x,train_y)

#print the model co-efficients and intercepts
print("Co-efficient of models: ",model.coef_)
print("Intercepts of the model: ",model.intercept_)

#predict on train set
train_pred = model.predict(train_x)
print("Prediction of train set ", train_pred)

#predict on test set
test_pred = model.predict(test_x)
print("Prediction on test set ",test_pred)

#evaluate both train set ans test set
rms_train = mean_squared_error(train_y,train_pred) ** (0.5)
print("RMS on train set ",rms_train)

rms_test = mean_squared_error(test_y,test_pred) ** (0.5)
print("RMS value of test set ",rms_test)
