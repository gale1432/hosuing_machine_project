import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
import joblib

def predict(data):
    lin_reg = joblib.load('linear_regression.plk')
    pipeline = joblib.load('pipeline.sav')
    pipelined_data = pipeline.fit_transform(data)
    print(len(pipelined_data))
    return lin_reg.predict(pipelined_data)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:, population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
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
        "INLAND",
        "NEAR OCEAN",
        "NEAR BAY",
        "ISLAND",
    ])
if st.button("Predict house price"):
    data = {'longitude': float(longitude), 'latitude': float(latitude), 'hma': float(hma), 'tr': float(tr),
            'tb': float(tb), 'pop': float(pop), 'house': float(house), 'mi': float(mi), 'op': op}
    df = pd.DataFrame([list(data.values())], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])
    result = predict(df)
    st.text(result[0])