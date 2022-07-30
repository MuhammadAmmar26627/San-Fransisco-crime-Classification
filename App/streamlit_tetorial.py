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
from util import *

adress_opt=get_location_names()
pdDistric_opt=get_pdDistric_names()
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
    st.write('***')
    col1,_,col2=st.columns([7,1,7])
    date=col1.date_input('date',value=datetime.date(2013,7,9),min_value=datetime.date(2013,7,9),max_value=datetime.date(2015,5,13))
    time=col2.time_input('time')
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
