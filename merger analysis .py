#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
file =pd.read_excel(r'C:\Users\umang\Desktop\SEMESTER 6\SOP\python.xlsx')
file.head()
#importing data from csv file containing sorted company data.

file.isnull().sum()
##checking for NA values

file.set_index('Brand Name',inplace=True)
# setting index as brand name

file.head()

from sklearn.model_selection import train_test_split 
x_var =file.drop('MA',axis=1) #all coloumns other than MA
y_var=file['MA']
## spliting the data into training data and test data
xTrain ,xValid ,yTrain , yValid = train_test_split(x_var,y_var,train_size=0.6,random_state=2)

from sklearn.linear_model import LogisticRegression
LogitModel= LogisticRegression()

LogitModel.fit(xTrain, yTrain)
#training the model on the data set

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as  plt
prediction = LogitModel.predict(xTrain)
accuracy_score(yTrain, prediction)
plt.scatter(y_test, predictions)
#accuracy of training model predictions


prediction2 = LogitModel.predict(xValid)
accuracy_score(yValid, prediction2)
#accuracy of validation model

from sklearn.metrics import confusion_matrix
confusion_matrix(yValid,prediction2)
# confusion matrix to see positive False and neagtives True

#ROC plot to see model accuracy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Log_ROC_auc =roc_auc_score(yValid,LogitModel.predict(xValid))
fpr,tpr,threshold= roc_curve(yValid,LogitModel.predict_proba(xValid)[:,1])
plt.figure()
plt.plot(fpr,tpr,label="Logit Model=%.02f)" % Log_ROC_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

##Logistic regression for the model
import statsmodels.api as sm

x_var=file.drop("MA",axis=1)
x_var =sm.add_constant(x_var)
xTrain2 ,xValid2 ,yTrain , yValid=train_test_split(x_var,y_var,train_size=0.6,random_state=2)
Logit_all= sm.Logit(yTrain,xTrain2)
result_All=Logit_all.fit()
print(result_All.summary())








