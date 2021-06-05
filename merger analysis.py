#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
file =pd.read_excel(r'C:\Users\umang\Desktop\SEMESTER 6\SOP\python.xlsx')
file.head()


# In[10]:


file.isnull().sum()


# In[11]:


file.set_index('Brand Name',inplace=True)


# In[12]:


file.head()


# In[14]:


from sklearn.model_selection import train_test_split 
x_var =file.drop('MA',axis=1)
y_var=file['MA']
xTrain ,xValid ,yTrain , yValid=train_test_split(x_var,y_var,train_size=0.6,random_state=2)


# In[15]:


from sklearn.linear_model import LogisticRegression
LogitModel= LogisticRegression()


# In[16]:


LogitModel.fit(xTrain, yTrain)


# In[26]:


from sklearn.metrics import accuracy_score
prediction = LogitModel.predict(xTrain)
accuracy_score(yTrain, prediction)


# In[30]:


prediction2 = LogitModel.predict(xValid)
accuracy_score(yValid, prediction2)


# In[32]:


from sklearn.metrics import confusion_matrix
confusion_matrix(yValid,prediction2)


# In[37]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Log_ROC_auc =roc_auc_score(yValid,LogitModel.predict(xValid))
fpr,tpr,threshold= roc_curve(yValid,LogitModel.predict_proba(xValid)[:,1])


# In[40]:


plt.figure()
plt.plot(fpr,tpr,label="Logit Model=%.02f)" % Log_ROC_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[52]:


import statsmodels.api as sm
x_age= file['age in ln']
x_age =sm.add_constant(x_age)
y_var=file['MA']


# In[53]:


xTrain1 ,xValid1 ,yTrain , yValid=train_test_split(x_age,y_var,train_size=0.6,random_state=2)


# In[54]:


Logit_age= sm.Logit(yTrain,xTrain)
result_Age=Logit_age.fit()
print(result_Age.summary())


# In[56]:


x_var=file.drop("MA",axis=1)
x_var =sm.add_constant(x_var)
xTrain2 ,xValid2 ,yTrain , yValid=train_test_split(x_var,y_var,train_size=0.6,random_state=2)


# In[59]:


Logit_all= sm.Logit(yTrain,xTrain2)
result_All=Logit_all.fit()
print(result_All.summary())


# In[60]:


x_red= x_var.drop(['capacity utilization','Leverage'],axis=1)


# In[65]:


x_red.head()


# In[66]:


xTrain3 ,xValid3 ,yTrain , yValid=train_test_split(x_red,y_var,train_size=0.6,random_state=2)
Logit_red= sm.Logit(yTrain,xTrain3)
result_red=Logit_red.fit()
print(result_red.summary())


# In[ ]:




