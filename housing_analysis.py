import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
import joblib

def predict(data):
    lin_reg = joblib.load('linear_regression.plk')
    pipeline = joblib.load('pipeline.sav')
    pipelined_data = pipeline.fit_transform(data)
    return lin_reg.predict(pipelined_data)

"""model = pickle.load(open('linear_regression.plk', 'rb'))
pipeline = pickle.load('pipeline.sav', 'rb')"""
st.title("Predecir precio de casa")
st.header("Variables")
col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input(label="Longitude")
    latitude = st.number_input(label="Latitude")
    hma = st.number_input(label="Housing Median Age")
    tr = st.number_input(label="Total Rooms")
    tb = st.number_input(label="Total Bedrooms")
with col2:
    pop = st.number_input(label="Population")
    house = st.number_input(label="Households")
    mi = st.number_input(label="Median Income")
    op = st.select_slider("Ocean Proximity", options=[
        "<1H OCEAN",
        "INLAND"
        "NEAR OCEAN"
        "NEAR BAY"
        "ISLAND"
    ])
#st.button("Predict house price")
if st.button("Predict house price"):
    result = predict(np.array([longitude, latitude, hma, tr, tb, pop, house, mi, op]))
    st.text(result[0])