# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:25:27 2017

@author: Raghav
"""

import numpy as np   
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing Dataset
dataset = pd.read_csv('Advertising.csv')
X = dataset.iloc[:, 1].values   # Matrix of features X
y = dataset.iloc[:, 4].values     # vector of labels 
y =np.c_[np.ones((200,0)), y]
X_b = np.c_[np.ones((200,1)), X] # Adding x0=1 to each instance

eta = 0.0001
n_iterations = 200
m = 200
theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * (X_b.T).dot(X_b.dot(theta) - y)
    theta = theta - (eta*gradients)
    
    
#Gradient Descent method
def plot_gradient_descent(theta,eta,n_iterations):
    for iteration in range(n_iterations):
        y_predict = X_b.dot(theta)
        plt.plot(X, y_predict, "-r")
        plt.plot(X,y,"b.")
        plt.axis([0,300,0,30])
        plt.ylabel("sales", rotation=0, fontsize=18)
        plt.title(r"$\eta = {}$".format(eta),fontsize=16)
    plt.show()
    print(theta)    

#Theta_best calculations using normal Equation

theta_best = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
X_test_b = np.c_[np.ones((40,1)), X_test] # Adding x0=1 to each instance
y_predict= X_test_b.dot(theta_best)  #predicting the test set
y_predict_train= X_train_b.dot(theta_best)  #predicting the train set


#Cost function or Mean Square Error calclations
from sklearn.metrics import mean_squared_error
mse= mean_squared_error(y_test, y_predict)


#Visualizing the Training set results
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, y_predict_train, color= 'blue')
plt.title('Sales vs Advertsising budget for TV (Training set)')
plt.xlabel('Advertising budget for TV')
plt.ylabel('Sales')
plt.show()


#Visualizing the Testing set results
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, y_predict_train, color= 'blue')
plt.title('Sales vs Advertsising budget for TV (Testing set)')
plt.xlabel('Advertising budget for TV')
plt.ylabel('Sales')
plt.show()
