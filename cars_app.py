
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load The Model And Pipeline
fl_pipeline = joblib.load('full_pipeline.pkl')
lr_model = joblib.load('poly_ridge_model.pkl')

# Load The Data
data = pd.read_csv("train.csv")

def inverse_log_transform(x):
    return np.expm1(x)

# Title 
st.title("🚗 Used Cars Price Prediction 🚗")
st.subheader("This App is built using a Linear Regression Model to predict the price of used cars.")

# Take Input From The User
Brand = st.selectbox("Brand", data["Brand"].unique())

Model = st.selectbox("Model", data["Model"].unique())

Fuel_Type = st.radio("Fuel Type", data["Fuel_Type"].unique())

Transmission = st.radio("Transmission", data["Transmission"].unique())

Power = st.slider(
    "Power", 
    int(data["Power"].min()),  
    int(data["Power"].max()),  
    int(data["Power"].median()) )

Engine = st.slider(
    "Engine", 
    int(data["Engine"].min()),  
    int(data["Engine"].max()),  
    int(data["Engine"].median()) )

Mileage_kmpl = Engine = st.slider(
    "Mileage(kmpl)", 
    int(data["Mileage(kmpl)"].min()),  
    int(data["Mileage(kmpl)"].max()),  
    int(data["Mileage(kmpl)"].median()) )

Kilometers_Driven = st.slider(
    "Kilometers Driven", 
    int(data["Kilometers_Driven"].min()),  
    int(data["Kilometers_Driven"].max()),  
    int(data["Kilometers_Driven"].median()) )

Location = st.selectbox("Location", data["Location"].unique())

Year = st.selectbox("Model Year", sorted(data["Year"].unique()))

Owner_Type = st.radio("Owner Type", data["Owner_Type"].unique())

Seats = st.selectbox("Number of Seats", sorted(data["Seats"].unique()))

Age = st.number_input("Age The Car", min_value=0, max_value=26, value=0, step=1)

# Store The Inputs as Dictionary
inputs = {"Brand" : Brand ,
          "Model" : Model ,
          "Fuel_Type" : Fuel_Type ,
          "Transmission" : Transmission ,
          "Power" : Power ,
          "Engine" : Engine ,
          "Mileage(kmpl)" : Mileage_kmpl ,
          "Kilometers_Driven" : Kilometers_Driven ,
          "Location" : Location , 
          "Year" : Year ,
          "Owner_Type" : Owner_Type ,
          "Seats" : Seats ,
          "Age" : Age}

# Transform the data into a DataFrame
features = pd.DataFrame(inputs, index=[0])  

# Pipeline transformation
features_prepared = fl_pipeline.transform(features)

# Predictions
prediction = lr_model.predict(features_prepared)[0]

#Appply Function
pred = inverse_log_transform(prediction)

# Display the prediction
st.subheader('Prediction Price This Car is:')
st.markdown(f'### $ {round(pred, 2)}')
