import pandas as pd
import numpy as np
import datetime
import time
import json

#takes in input the dataset used as testSet and the datset with results
def evaluate(test, result):
    merged = merge_for_evaluation(test, result)
    num = (np.abs(merged['NumberOfSales_x']- merged['NumberOfSales_y'])).groupby(test['Region']).sum()
    den = merged['NumberOfSales_x'].groupby(test['Region']).sum()
    err_region = np.array(num/den)
    n_regions = len(merged['Region'].unique())
    err = np.sum(err_region, axis=0)/n_regions
    return err

def merge_for_evaluation(test, result):
    target =test.groupby(['StoreID','Month'], as_index=False)['NumberOfSales'].sum()
    region =test.groupby(['StoreID','Month'], as_index=False)['Region'].mean()
    target = pd.merge(target, region,   how='inner', on=['StoreID', 'Month'])
    merged = pd.merge(target, result, how='inner', on=['StoreID', 'Month'])
    return merged
