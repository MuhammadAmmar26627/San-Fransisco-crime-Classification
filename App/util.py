import json
import joblib
import pandas as pd
import numpy as np
import datetime
import os
__location=None
__pdDistric=None
__data_columns=None
__model=None

def predict(date,time,pdDistrict,adress,X_cor,Y_cor,):
    
    x_pre=np.zeros(np.shape(__data_columns))
    print((type(__data_columns)))
    if adress!='other':
        adress_index=np.where(__data_columns==adress)[0][0]
        x_pre[adress_index]=1
    if pdDistrict!='PARK':
        pdDistrict_index=np.where(__data_columns==pdDistrict)[0][0]
        x_pre[pdDistrict_index]=1
    
    if 'Block' in adress:
        block_index=np.where(__data_columns=='Block')[0][0]
        x_pre[block_index]=1
    if '/' in adress:
        int_index=np.where(__data_columns=='Intersect')[0][0]
        x_pre[int_index]=1
    
    x_pre[1]=float(X_cor)
    x_pre[2]=float(Y_cor)
    date=pd.to_datetime(date)
    year=date.year
    x_pre[3]=year
    month=date.month
    x_pre[4]=month
    day=date.day
    x_pre[5]=day
    hr=time.hour
    x_pre[6]=hr
    minutes=time.minute
    x_pre[7]=minutes
    dayofweek=date.weekday()
    x_pre[0]=dayofweek
    results=['ASSAULT','DRUG/NARCOTIC','LARCENY/THEFT','NON-CRIMINAL','OTHER OFFENSES']
    return results[__model.predict([x_pre])[0]]

def get_location_names():
    load_saved_artifacts()
    
    return __location
def get_pdDistric_names():
    return __pdDistric
def get_data_columns():
    return __data_columns
def load_saved_artifacts():
    global __data_columns
    global __location
    global __pdDistric
    global __model
    with open('column.json','rb') as f:
        __data_columns=json.load(f)['data_columns']
        __location=__data_columns[10:-9]
        __location=[i.replace(' ','_') for i in __location ]
        __pdDistric=__data_columns[-9:]
        __data_columns=np.array(__data_columns)
    __model=joblib.load('bag_model56.pkl')
if __name__=='__main__':
    load_saved_artifacts()