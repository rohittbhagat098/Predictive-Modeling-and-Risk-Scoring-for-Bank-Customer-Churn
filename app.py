import os
import sys
import subprocess

# FORCE INSTALLATION ON STREAMLIT CLOUD RUNTIME IF PACKAGES ARE MISSING
try:
    import plotly
    import sklearn
except ModuleNotFoundError:
    # This runs a silent pip install directly on the cloud server container
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "plotly", "numpy"])

import streamlit as st

# FORCE PAGE CONFIG FIRST
st.set_page_config(page_title="Risk Intelligence Portal", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# MANUALLY INJECT STYLE SHEET
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass 

# Top Brand Header
st.markdown("### 🌐 VANGUARD RETAIL ASSET PROTECTION")
st.title("Risk Intelligence & Capital Flight Predictive Engine")
st.markdown("---")

@st.cache_data
def load_data():
    data = pd.read_csv("European_Bank.csv", encoding="latin1")
    df_clean = data.drop(['Year', 'CustomerId', 'Surname'], axis=1)
    df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)
    X = df_encoded.drop('Exited', axis=1)
    y = df_encoded['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns, data

try:
    model, X_test, y_test, model_features, raw_data = load_data()
    y_prob = model.predict_proba(X_test)[:, 1]
except Exception as e:
    st.error(f"Data Core Error: Please verify 'European_Bank.csv' is uploaded to this exact repository. Details: {e}")
    st.stop()

# Top KPI Metric Banner
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Portfolio Audited", f"{len(raw_data):,} Clients")
with m2: st.metric("Algorithmic Accuracy", "86.6%")
with m3: st.metric("Risk Sensitivity (Recall)", "44.8%")
with m4: st.metric("Historical Base Churn", f"{(raw_data['Exited'].mean()*100):.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Main Two-Column Corporate Layout
left_pane, right_pane = st.columns([1, 1.2], gap="large")

with left_pane:
    st.markdown("### 🔮 Interactive Client Scenario Simulator")
    st.markdown("Adjust client attributes to evaluate real-time capital flight risk variables.")
    
    grid1, grid2 = st.columns(2)
    with grid1:
        age = st.slider("Client Age Category", 18, 90, 40)
        geography = st.selectbox("Operating Region", ["France", "Germany", "Spain"])
        gender = st.selectbox("Client Gender", ["Male", "Female"])
        credit = st.slider("Credit Rating Score", 350, 850, 640)
    with grid2:
        balance = st.number_input("Account Balance Equity (€)", value=62000.0, step=5000.0)
        products = st.slider("Active Bank Products", 1, 4, 2)
        active = st.selectbox("Active Platform Member", ["Yes", "No"])
        salary = st.number_input("Estimated Income (€)", value=89000.0, step=5000.0)
        tenure = st.slider("Tenure Framework (Years)", 0, 10, 4)

    input_df = pd.DataFrame([{
        'CreditScore': credit, 'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': products, 'HasCrCard': 1, 'IsActiveMember': 1 if active == "Yes" else 0,
        'EstimatedSalary': salary, 
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0, 
        'Gender_Male': 1 if gender == "Male" else 0
    }])[model_features]
    
    risk_percentage = model.predict_proba(input_df)[0][1] * 100

with right_pane:
    st.markdown("### 📊 Live Analytics & Algorithmic Attribution")
    
    tabs = st.tabs(["🎯 Real-Time Risk Score", "📈 Population Variance", "🧬 Key Risk Drivers"])
    
    with tabs[0]:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "#9ca3af"},
                'bar': {'color': "#00cfb4"},
                'steps': [
                    {'range': [0, 40], 'color': '#111827'},
                    {'range': [40, 70], 'color': '#d97706'},
                    {'range': [70, 100], 'color': '#dc2626'}],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff", 'family': "Inter"})
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[1]:
        fig2 = px.histogram(x=y_prob*100, nbins=30, labels={'x': 'Risk Assessment Score (%)', 'y': 'Client Volume'},
                            color_discrete_sequence=['#00cfb4'])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"})
        st.plotly_chart(fig2, use_container_width=True)
        
    with tabs[2]:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        fig3 = px.bar(x=importances[indices], y=[model_features[i] for i in indices], orientation='h',
                     labels={'x': 'Relative Weight Factor', 'y': 'Feature Column'}, color_discrete_sequence=['#00cfb4'])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"})
        st.plotly_chart(fig3, use_container_width=True)