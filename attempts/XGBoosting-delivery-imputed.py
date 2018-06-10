
# coding: utf-8

# # Gradient Boosting

# In[41]:


import pandas as pd
import numpy as np
from aux_fun import my_eval, my_grid_search_cv
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble

train_data = pd.read_csv('./data/train_imputed.csv')
prediction_data = pd.read_csv('./data/test_imputed.csv')

#train_data=train_data[0:100]
#prediction_data=prediction_data[0:100]

prediction_data=prediction_data.drop('WindDirDegrees', axis=1)
# In[42]:


#train_data.head()


# # Dropping problems

# In[43]:


X=train_data.drop(['NumberOfSales','WindDirDegrees','NumberOfCustomers'], axis=1)


# In[44]:


y=train_data['NumberOfSales']


# # Holdout

# In[45]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Here we go with Pizza Boosting

# In[ ]:

print("Sto andando")
params_array = {'max_depth' : 10, 'n_estimators': 1000}
xgboost=xgb.XGBRegressor(**params_array)
xgboost.fit(X,y)
y_pred=xgboost.predict(prediction_data)

result = pd.DataFrame(prediction_data['StoreID'])
result['Month']=prediction_data['Month']
result['NumberOfSales'] = y_pred
#Group by Month
result = result.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()

result.to_csv('./results/predictions_imputed.csv', index=False)


# In[ ]:


#print(results)
