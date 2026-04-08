import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model.h5')

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoded_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

credit_score = st.slider('Credit Score', 300, 850, 619)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.slider('Age', 18, 92, 42)
tenure = st.slider('Tenure (years)', 0, 10, 2)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.selectbox('Is Active Member', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=101348.88)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    input_data['gender'] = label_encoder_gender.transform(input_data['Gender'])

    geo_encoded = one_hot_encoded_geo.transform(input_data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoded_geo.get_feature_names_out())

    input_df = pd.concat([input_data.drop(['Geography', 'Gender'], axis=1), geo_encoded_df], axis=1)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    prob = prediction[0][0]

    st.subheader('Prediction Result')
    st.write(f'Churn Probability: **{prob:.2%}**')
    if prob > 0.5:
        st.error('This customer is likely to churn.')
    else:
        st.success('This customer is not likely to churn.')
