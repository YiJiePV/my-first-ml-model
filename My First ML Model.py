#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib


# In[11]:


df = pd.read_csv('~/Downloads/archive/Melbourne_housing_FULL.csv') #import dataset from computer


# In[22]:


df.head(n=10) #preview dataset


# In[16]:


del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']
#delete unneeded columns


# In[19]:


df.dropna(axis=0, how='any', subset=None, inplace=True) #drop rows with missing values


# In[21]:


features_df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])


# In[25]:


features_df.head(n=10)


# In[24]:


del features_df['Price'] #don't need price column (will be dependent variable)


# In[30]:


x = features_df.values #array of independent variables
y = df['Price'].values #array of dependent variables


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) #split dataset by standard 70/30


# In[41]:


model = ensemble.GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=5, min_samples_split=4, min_samples_leaf=6, max_features=0.6, loss='huber')
#hyperparams:
#n_estimators == how many decision tree to build
#learning_rate == rate at which additional decision trees influence overall prediction
#max_depth == max number of layers (depth) for each decision tree
#min_samples_split == min number of samples required to execute new binary split
#min_samples_leaf == min number of samples that must appear in each child (leaf) before new branch can be implemented
#max_features == total num of features presented to model when determining best split
#loss == model's error rate


# In[42]:


model.fit(x_train, y_train) #start training model


# In[39]:


joblib.dump(model, 'house_trained_model.pkl') #save training model as file


# In[43]:


mse = mean_absolute_error(y_train, model.predict(x_train)) #calculate mean absolute error of model
print("Training Set Mean Absolute Error: %.2f" % mse)


# In[44]:


mse = mean_absolute_error(y_test, model.predict(x_test)) #same thing but with test data
print("Test Set Mean Absolute Error: %.2f" % mse)

