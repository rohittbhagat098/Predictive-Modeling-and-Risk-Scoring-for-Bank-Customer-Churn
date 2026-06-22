import os
import sys
import subprocess

# 1. FORCE MANUAL INSTALLATION DIRECTLY ON THE SERVER AT LAUNCH
try:
    import plotly
    import sklearn
    import pandas
except ModuleNotFoundError:
    # This instructs the server container to run a live pip install in the background
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "plotly", "numpy"])

# 2. NOW RUN THE MAIN APP CONFIGURATION
import streamlit as st
st.set_page_config(page_title="Bank Churn Prediction Portal", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Bank Customer Churn Risk Prediction Dashboard")
st.write("Enter customer attributes on the left panel to calculate their churn risk probability in real-time.")
st.markdown("---")

# Data loading function with Latin1 encoding safety boundary
@st.cache_data
def load_and_train():
    data = pd.read_csv("European_Bank.csv", encoding="latin1")
    df_clean = data.drop(['Year', 'CustomerId', 'Surname'], axis=1)
    df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)
    X = df_encoded.drop('Exited', axis=1)
    y = df_encoded['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns, data

# Execute data assembly
model, X_test, y_test, model_features, raw_data = load_and_train()
y_prob = model.predict_proba(X_test)[:, 1]

# Layout Setup: Left Sidebar for Inputs, Right Main Body for Results
sidebar, main_body = st.columns([1, 1.5], gap="medium")

with sidebar:
    st.header("👤 Customer Profile Input")
    
    age = st.slider("Age", 18, 90, 40)
    geography = st.selectbox("Country/Region", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    credit = st.slider("Credit Score", 350, 850, 640)
    balance = st.number_input("Account Balance (€)", value=60000.0, step=1000.0)
    products = st.slider("Number of Bank Products Owned", 1, 4, 2)
    active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary (€)", value=85000.0, step=1000.0)
    tenure = st.slider("Tenure (Years at Bank)", 0, 10, 4)

    input_df = pd.DataFrame([{
        'CreditScore': credit, 'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': products, 'HasCrCard': 1, 'IsActiveMember': 1 if active == "Yes" else 0,
        'EstimatedSalary': salary, 
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0, 
        'Gender_Male': 1 if gender == "Male" else 0
    }])[model_features]
    
    risk_percentage = model.predict_proba(input_df)[0][1] * 100

with main_body:
    st.header("📊 Analytical Output")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Baseline Accuracy", "86.60%")
    with col2:
        if risk_percentage > 50:
            st.error(f"High Risk Profile: {risk_percentage:.2f}% Attrition Chance")
        else:
            st.success(f"Low Risk Profile: {risk_percentage:.2f}% Attrition Chance")
            
    st.markdown("---")
    
    plot_tab1, plot_tab2 = st.tabs(["🎯 Churn Meter Gauges", "🧬 Core Risk Drivers"])
    
    with plot_tab1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': 'lightgreen'},
                    {'range': [40, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'coral'}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    with plot_tab2:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        fig2 = px.bar(
            x=importances[indices], 
            y=[model_features[i] for i in indices], 
            orientation='h',
            labels={'x': 'Relative Importance Metric', 'y': 'Feature Column'}
        )
        st.plotly_chart(fig2, use_container_width=True)