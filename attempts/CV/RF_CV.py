
# coding: utf-8

# # Gradient Boosting

# In[4]:


import pandas as pd
import numpy as np

import xgboost as xgb

train_data = pd.read_csv('./data/train_imputed.csv')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.model_selection import KFold
from aux_fun import my_eval, r2_month, compute_kf
from sklearn.ensemble import RandomForestRegressor


# # Dropping problems

# In[5]:


X=train_data.drop(['NumberOfSales','WindDirDegrees','NumberOfCustomers'], axis=1)
y=train_data['NumberOfSales']


# # Here we go with Pizza Boosting

# In[6]:

print("Sto andando")
n_folds=10
error=0
r2=0
r2_month=0
ind_params = {'n_estimators': 30}
model = RandomForestRegressor(**ind_params)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
error,r2,r2_month=compute_kf(kf,X,y,False,model,n_folds)


# In[7]:


print("Error:")
print(error)
print("R2_month:")
print(r2)
print("R2:")
print(r2_month)
