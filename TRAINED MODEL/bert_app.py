import streamlit as st
import pandas as pd
import numpy as np

# Core packages for text processing.
import string
import re
import warnings
import time
import datetime
import pickle

pickl_model = pickle.load(open('linear_pred.pkl', 'rb'))

# Setting some options for general use.
import os
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# -----------------------------------------------------------------

st.markdown("""
<style>
div.stButton > button:first-child {
background-color: #00cc00;
color:black;
font-size:15px;
height:2.7em;
width:20em;
border-radius:10px 10px 10px 10px;}
</style>
    """,
            unsafe_allow_html=True
            )

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTI8DRxgEP4PMaChfWJQKulfwMWdF486bB0SF0ZHXkgS5z4gc2Jd7EGKC8-gjjKWNxEUlQ&usqp=CAU")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# -------------------------------------------------------------


st.title('Diabetes Prediction using ML')
##getting the input 
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col3:
    BloodPressure = st.text_input('Blood pressure value')

with col1:
    SkinThickness = st.text_input('skin thickness value')

with col2:
    Insulin = st.text_input('Insulin level')

with col3:
    BMI = st.text_input('BMI Value')

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree function value')

with col2:
    Age = st.text_input('Age of the person')



diab_diagnosis = ' '
diab_prediction = pickl_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

Pregnancies = input('Number of pregnancies')
Glucose = input('Glucose Level')
BloodPressure = input('Blood Pressure Value')
SkinThickness = input('Skin Thickness Value')
Insulin = input('Insulin Level')
BMI = input('BMI value')
DiabetesPedigreeFunction = input('Diabete Pedigree Function Value')
Age = input('Age of the person')

diab_diagnosis = ' '

if st.button('Diabetes test result'):
    diab_prediction = pickl_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])


if (diab_prediction[0]==1):
    diab_diagnosis = 'The person is diabetic'
else:
    diab_diagnosis = 'The person is not diabetic'

st.success(diab_diagnosis)

