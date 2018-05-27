import pandas as pd
import numpy as np
import datetime
import time

#takes in input the dataset used as testSet and the datset with results
def evaluate(test, result):

    merged = merge_for_evaluation(test, result)
    num = (np.abs(merged['NumberOfSales_x']- merged['NumberOfSales_y'])).groupby(test['Region']).sum()
    #print(num)
    den = merged['NumberOfSales_x'].groupby(test['Region']).sum()
    #print(den)
    err_region = np.array(num/den)
    n_regions = len(merged['Region'].unique())
    err = np.sum(err_region, axis=0)/n_regions
    return err

def merge_for_evaluation(test, result):
    test['Date']= pd.to_datetime(test['Date'])
    test['Month']=pd.DatetimeIndex(test['Date']).month
    target =test.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()
    region =test.groupby(['StoreID','Month'], as_index=False)['Region'].mean()
    target = pd.merge(target, region,   how='inner', on=['StoreID', 'Month'])
    merged = pd.merge(target, result, how='inner', on=['StoreID', 'Month'])
    return merged
