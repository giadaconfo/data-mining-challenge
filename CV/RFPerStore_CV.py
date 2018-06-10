
# coding: utf-8

# # A Random forest per store

# In[1]:


import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from aux_fun import my_eval, r2_month, compute_kf

import json


# In[2]:


df = pd.read_csv('./data/train_imputed.csv')
Y = df[['StoreID','NumberOfSales']]
X = df.drop(df[['NumberOfSales','NumberOfCustomers', 'WindDirDegrees']], axis=1)


# In[3]:


stores=X['StoreID'].unique().astype(int)


# # Model

# In[5]:
print("Sto andando")

n_folds=10
error=0
r2=0
r2_month=0
i=0
y_pred=[]
ind_params = {'n_estimators': 30}
model = RandomForestRegressor(**ind_params)

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
for store in stores:
    if i%100==0:
        print("Eval: "+str(i))

    #forming dataset
    y_store=Y[Y['StoreID']==store].drop('StoreID',axis=1)
    X_store=X[X['StoreID']==store]


    error_cv,r2_cv,r2_month_cv=compute_kf(kf,X_store,y_store,False,model,n_folds)
    error+=error_cv
    r2+=r2_cv
    r2_month+=r2_month_cv
    i+=1
error=error/len(stores)
r2=r2/len(stores)
r2_month=r2_month/len(stores)


# In[9]:


print("Error:")
print(error)
print("R2_month:")
print(r2)
print("R2:")
print(r2_month)
