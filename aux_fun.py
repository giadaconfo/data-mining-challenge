import pandas as pd
import datetime
import time

def evaluate(test, result):
    merged = pd.merge(test, result, how='inner', on=[['StoreID', 'Month']])
    den = test['NumberOfSales'].groupby(test['Region']).sum()
    #num = (test['NumberOfSales']-result['NumberOfSales']).
    err = num/den
    return err


test = pd.read_csv('/home/giada/github/data-mining-challenge/data/train_imputed.csv')
results = pd.read_csv('/home/giada/github/data-mining-challenge/data/sample_submission.csv')
results.head()
test['Date']= pd.to_datetime(test['Date'])
test['Month']=pd.DatetimeIndex(test['Date']).month
test['Month']
target = pd.DataFrame(test[['StoreID','Month','NumberOfSales']])
tot_sale_month_store = pd.DataFrame(target.groupby(['StoreID','Month'])['NumberOfSales'].sum())
tot_sale_month_store
#TODO Save in a correct dataframe

evaluate(tot_sale_month_store, results)
