import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import datetime
import pandas as pd
# from util import *


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
    # with open('column.json','rb') as f:
        # __data_columns=json.load(f)['data_columns']
    __data_columns=["DayOfWeek", "X", "Y", "year", "month", "day", "hour", "minute", "Block", "Intersect", "0 Block of 6TH ST", "0 Block of JONES ST", "0 Block of OFARRELL ST", "0 Block of PHELAN AV", "0 Block of POWELL ST", "0 Block of TURK ST", "0 Block of UNITEDNATIONS PZ", "100 Block of 6TH ST", "100 Block of EDDY ST", "100 Block of GOLDEN GATE AV", "100 Block of LEAVENWORTH ST", "100 Block of OFARRELL ST", "100 Block of POWELL ST", "100 Block of TAYLOR ST", "100 Block of TURK ST", "1000 Block of MARKET ST", "1000 Block of POTRERO AV", "1100 Block of MARKET ST", "1200 Block of MARKET ST", "1400 Block of PHELPS ST", "1600 Block of THE EMBARCADERONORTH ST", "16TH ST / MISSION ST", "200 Block of EDDY ST", "200 Block of INTERSTATE80 HY", "200 Block of TURK ST", "2000 Block of MARKET ST", "2000 Block of MISSION ST", "2300 Block of 16TH ST", "300 Block of EDDY ST", "300 Block of ELLIS ST", "3000 Block of 16TH ST", "3200 Block of 20TH AV", "3300 Block of MISSION ST", "400 Block of CASTRO ST", "400 Block of EDDY ST", "400 Block of ELLIS ST", "400 Block of JONES ST", "400 Block of OFARRELL ST", "500 Block of JOHNFKENNEDY DR", "600 Block of VALENCIA ST", "700 Block of MARKET ST", "700 Block of STANYAN ST", "800 Block of BRYANT ST", "800 Block of MARKET ST", "900 Block of MARKET ST", "900 Block of POTRERO AV", "ELLIS ST / JONES ST", "MISSION ST / 16TH ST", "TURK ST / LEAVENWORTH ST", "TURK ST / TAYLOR ST", "BAYVIEW", "CENTRAL", "INGLESIDE", "MISSION", "NORTHERN", "RICHMOND", "SOUTHERN", "TARAVAL", "TENDERLOIN"]
    print(__data_columns)
    __location=__data_columns[10:-9]
    __location=[i.replace(' ','_') for i in __location ]
    __pdDistric=__data_columns[-9:]
    __data_columns=np.array(__data_columns)
#     model=open(r'bag_model56.pkl','rb')
    __model = pickle.load(open(r'bag_model56.pkl, 'rb'))
#     __model=joblib.load(model)



@st.cache
def loaddf():
    df1=pd.read_csv(r'sample.csv',parse_dates=['Dates'],index_col='Dates',usecols=['Category','Dates','lat','lon'])
    df=pd.get_dummies(df1.Category)
    df=pd.concat([df,df1],axis=1)
    return df
rad=option_menu(None,['Home','About'],icons=['house','book'],menu_icon='cast',orientation='horizontal')


if rad=='Home':
    st.write('''# San Francisco Crime Classification
This app predict top 5 crime in *San Francisco*. 
''')
#     adress_opt=get_location_names()
#     pdDistric_opt=get_pdDistric_names()
    st.write('***')
    col1,_,col2=st.columns([7,1,7])
    date=col1.date_input('date',value=datetime.date(2013,7,9),min_value=datetime.date(2013,7,9),max_value=datetime.date(2015,5,13))
    time=col2.time_input('time')
    
    
    adress_opt=get_location_names()
    pdDistric_opt=get_pdDistric_names()
    
    
    col1,_,col2=st.columns([7,1,7])
    PdDistric=col1.selectbox('PdDistric',pdDistric_opt)
    Adress=col2.selectbox('Adress',adress_opt)
    Adress=Adress.replace('_',' ')
    col1,_,col2=st.columns([7,1,7])
    X_cor = col1.number_input('Lon',value=-122.3655654,min_value=-122.5136421,max_value=-122.3655654)
    Y_cor = col2.number_input('Lat',value=37.70803052,min_value=37.70803052,max_value=37.81992346)
    st.write('***')
    col1,_,col2=st.columns([7,1,7])
    if col1.button('Predict'):
        Crime=predict(date,time, PdDistric, Adress, X_cor, Y_cor)
        col2.subheader(f'Crime Predicted is: {Crime}')
    st.markdown(""" <style> div.stButton > button:first-child { background-color: rgb(204, 49, 49);width:8.5cm } </style>""", unsafe_allow_html=True)
    
    st.write('***')
    
    
    df1=loaddf()
    # if st.checkbox('Show DataSet'):
    #     st.write(df1)
    
    col1,col2,col3=st.columns([1,1,1])
    start_date=col1.date_input('End _date',value=datetime.date(2013,7,9),min_value=datetime.date(2013,7,9),max_value=datetime.date(2015,5,13))
    End_date=col2.date_input('End_date',value=datetime.date(2015,5,13),max_value=datetime.date(2015,5,13),min_value=datetime.date(2013,7,9))
    option = col3.selectbox(
     'Ploting Frequecy?',
     ('SM', 'M','Y','H','D','W'))
    df=df1.loc[str(start_date):str(End_date)]
    st.map(df,zoom=10)
    
    
    st.line_chart(df[df.columns[:-2]].resample(option).sum())
    st.bar_chart(df[df.columns[:-2]].resample(option).sum())

    
    st.write('***')
 
if rad=='About':
    
    st.header('Problem Statement')
    st.write('To examine the specific problem, we will apply a full Data Science life cycle composed of the following steps:')

    list_data=['Data Wrangling to audit the quality of the data and perform all the necessary actions to clean the dataset.',
    'Data Exploration for understanding the variables and create intuition on the data.',
    'Feature Engineering to create additional variables from the existing.',
    'Data Normalization and Data Transformation for preparing the dataset for the learning algorithms (if needed).',
    'Training / Testing data creation to evaluate the performance of our models and fine-tune their hyperparameters.',
    'Model selection and evaluation. This will be the final goal; creating a model that predicts the probability of each type of crime based on the location and the date.']
        
    for i in list_data:
            st.markdown('* ' + i)
        
    st.header('Data fields')
    
    list_data=[
    'Dates: timestamp of the crime incident',
    'Category: category of the crime incident',
    'Descript: detailed description of the crime incident',
    'DayOfWeek: the day of the week',
    'PdDistrict: name of the Police Department District',
    'Resolution: how the crime incident was resolve',
    'Address: the approximate street address of the crime incident',
    'X: Longitude',
    'Y: Latitude']
    for i in list_data:
        st.markdown('* ' + i)
    
    st.header('Model selected is Decision Trees with accoracy 84%')
    
    
    
    
    st.subheader('Crime Rate')
    st.image([r'Graph\crime.png'])

    # st.subheader('Crime Rate')
    st.image([r'Graph\dataplot.png'])

    col1,col2=st.columns([1,1])
    col1.subheader('confusion matrix')
    col1.image([r'Graph\Predict56.png'])
    col2.subheader('correlation matrix')
    col2.image([r'Graph\heatmap_corr.png'])
