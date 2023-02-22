import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle

df = pickle.load(open('df1.pkl','rb'))
model = pickle.load(open('lr1.pkl','rb'))

st.title('Job Placement Prediction')
st.header('Fill the details to predict the Placement Status')

# Features
gender = st.selectbox('Gender',df['gender'].unique())
ssc_percentage = st.number_input('ssc_percentage')
ssc_board = st.selectbox('ssc_board',df['ssc_board'].unique())
hsc_percentage = st.number_input('hsc_percentage')
hsc_board = st.selectbox('hsc_board',df['hsc_board'].unique())
hsc_subject = st.selectbox('hsc_subject',df['hsc_subject'].unique())
degree_percentage = st.number_input('degree_percentage')
undergrad_degree = st.selectbox('undergrad_degree',df['undergrad_degree'].unique())
work_experience = st.selectbox('work_experience',df['work_experience'].unique())
emp_test_percentage = st.number_input('emp_test_percentage')
specialisation = st.selectbox('specialisation',df['specialisation'].unique())
mba_percent = st.number_input('mba_percent')


if st.button('Predict Placement Status'):
    test_data = np.array([gender,ssc_percentage,ssc_board,hsc_percentage,hsc_board,
                            hsc_subject,degree_percentage,undergrad_degree,work_experience,
                            emp_test_percentage,specialisation,mba_percent])
    test_data = test_data.reshape([1,12])

    st.success(model.predict(test_data)[0])

# To start the server - streamlit run ml_app1.py
# Stop Server - Ctrl + C

