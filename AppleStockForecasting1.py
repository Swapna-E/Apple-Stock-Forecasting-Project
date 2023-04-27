# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 00:07:08 2023

@author: admin
"""

import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pickle import dump
from pickle import load

apple_stk=pd.read_csv("AAPL.csv")

a = apple_stk
size = int(len(a) * 0.75)
Train, Test = a[0:size], a[size:len(a)]

X = apple_stk['Close'].values
size = int(len(X) * 0.75)
train, test = X[0:size], X[size:len(X)]

hwe_model_mul_sea = ExponentialSmoothing(Train["Close"],seasonal="mul",trend="add",seasonal_periods=252).fit() 
pred_hwe_mul_sea_train = hwe_model_mul_sea.predict(start = Train.index[0],end = Train.index[-1])
pred_hwe_mul_sea_test = hwe_model_mul_sea.predict(start = Test.index[0],end = Test.index[-1])

actualfig=plt.figure(figsize=(20,8))
plt.plot(apple_stk.Close, color='red',label='Actual')
plt.plot(pred_hwe_mul_sea_train, color='green',label='Train Predicted')
plt.plot(pred_hwe_mul_sea_test, color='blue',label='Test Predicted')
plt.title('apple stock close value prediction by Holts Winter multiplicative Seasonality Method', fontsize=16)
plt.ylabel('Close', fontsize=16)
plt.legend()
plt.show()


st.title('Model Deployment: Apple Stock Forecasting')

st.pyplot(actualfig)

st.sidebar.header('User Input Parameters')

Days = st.sidebar.number_input("Number of days for forecasting",min_value=1,max_value=1000,step=1)

st.subheader('User Input parameters')
st.write(Days)

import pickle

with open(file='Final_Model.pkl',mode='rb') as f:
    model = pickle.load(f)
    
 
    
result = model.forecast(Days)

#res = pd.Series(data = result)

#st.write(res)

st.write("Forecasted Results")

def load_result():
    
    #res = pd.Series(result , index=[1])
    
    df1 = pd.DataFrame(result, columns=['Apple_Stock_Close_Value']) 
    
    return df1

df=load_result()
#df.index = range(len(df.index))

st.write(df)

fig =plt.figure(figsize=(16,8))
plt.plot(df['Apple_Stock_Close_Value'], color='green',label='Forecasted Values')
plt.title('Apple Stock Forecasting Results', fontsize=18)
plt.ylabel('Forecasted Close Values', fontsize=14)
plt.xlabel('Number of Forecasted Days', fontsize=14)
plt.legend()
plt.show()
st.pyplot(fig)

  
actualfig1=plt.figure(figsize=(20,8))
plt.plot(apple_stk.Close, color='red',label='Actual')
plt.plot(df['Apple_Stock_Close_Value'], color='blue',label='Forecasted Values')
plt.title('apple stock forecasted close value with actual close value', fontsize=16)
plt.ylabel('Close', fontsize=16)
plt.legend()
plt.show()
st.pyplot(actualfig1)
   