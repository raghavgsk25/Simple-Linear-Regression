# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:17:29 2017

@author: Raghav
"""

import numpy as np   #as is used for shortcut, so ow np can be used to access the numpy library. Numpy library-mathmatical operations
import matplotlib.pyplot as plt #pyplot is sub library of former, used to plot graphs
import pandas as pd #pd library used to import and manage datasets

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values   # Matrix of features X
y = dataset.iloc[:,1].values     # vector of labels


#Taking care of missing data
'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis =0)
imputer = imputer.fit(X[:, 1:3]) #fitting inputer object to our matrix X 1 and 2 coulumn only considered
X[:, 1:3] = imputer.transform(X[:, 1:3])''' 

# Encoding categorical data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features =[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

#Splitting the dataset into the training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 42)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.intercept_, regressor.coef_

#Predicting the Test set Results
y_pred = regressor.predict(X_test)


#Visualizing the Training set results
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary in Dollars')
plt.show()


#Visualizing the Testing set results
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train,regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary in Dollars')
plt.show()


