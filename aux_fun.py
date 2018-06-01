import pandas as pd
import numpy as np
import datetime
import time
import json
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import sklearn

def my_eval(X_test, y_test, y_pred):
    X_test.loc[:,'NumberOfSales']= y_test
    result = pd.DataFrame(X_test['StoreID'])
    result.loc[:,'Month'] = X_test['Month']
    result.loc[:,'NumberOfSales'] = y_pred
    result =result.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()
    merged = merge_for_evaluation(X_test, result)
    num = (np.abs(merged['NumberOfSales_x']- merged['NumberOfSales_y'])).groupby(merged['Region']).sum()
    den = merged['NumberOfSales_x'].groupby(merged['Region']).sum()
    err_region = np.divide(num,den)
    n_regions = len(merged['Region'].unique())
    err = np.divide(np.sum(err_region, axis=0),n_regions)
    return err

def merge_for_evaluation(test, result):
    target =test.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()
    region =test.groupby(['StoreID','Month'], as_index=False)['Region'].mean()
    target = pd.merge(target, region,   how='inner', on=['StoreID', 'Month'])
    merged = pd.merge(target, result, how='inner', on=['StoreID', 'Month'])
    return merged

def r2_month(X_test, y_test, y_pred ):
    test =X_test.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()
    result = pd.DataFrame(X_test['StoreID'])
    result.loc[:,'Month'] = X_test['Month']
    result.loc[:,'NumberOfSales'] = y_pred
    #Group by Month
    result =result.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()
    r2_month = r2_score(test['NumberOfSales'], result['NumberOfSales'])
    return r2_month

'''
model: object of the model we want to use
params_array: array of dictionaries of values we want to test for Parameters
X_mat: matrix X
y: target
n_folds: folds for cv
'''
def my_grid_search_cv(model, params_array, X_mat, y, n_folds=5):
    res = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for param in params_array:
        eval_array =[]
        R2_array = []
        R2_month_array =[]
        model_type = type(model)
        model = model_type(**param)
        for train_index, test_index in kf.split(X_mat):
            X_traincv, X_testcv = X_mat.iloc[train_index], X_mat.iloc[test_index]
            y_traincv, y_testcv = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_traincv, y_traincv)
            y_pred = model.predict(X_testcv)
            evaluation =my_eval(X_testcv,y_testcv,y_pred)
            r2 = r2_score(y_testcv,y_pred)
            r2_m = r2_month(X_testcv, y_testcv, y_pred)
            eval_array.append(evaluation)
            R2_array.append(r2)
            R2_month_array.append(r2_m)
        avg_error = np.divide(np.array(eval_array).sum(), n_folds)
        avg_r2 = np.divide(np.array(R2_array).sum(),n_folds)
        avg_r2_month = np.divide(np.array(R2_month_array).sum(),n_folds)
        row =[model, n_folds, param, avg_error, avg_r2, avg_r2_month]
        res.append(row)
    results = pd.DataFrame(res,columns=['Method', 'Folds', 'Parameters', 'Eval_test', 'R2', 'R2_month'])
    return results
