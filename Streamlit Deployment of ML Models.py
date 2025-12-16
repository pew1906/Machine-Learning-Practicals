# Assignment 09: Streamlit Deployment of ML Models

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Classification - Logistic vs Random Forest", layout="wide")
st.title(" Wine Classification using Logistic Regression & Random Forest")
st.markdown("This Streamlit app compares **Logistic Regression** and **Random Forest** on the Wine dataset using SMOTE balancing.")

DATA_PATH = r"C:\Users\0555\Downloads\wine-class.csv"  # your provided path
st.sidebar.header("Dataset Info")
st.sidebar.write(f"Using dataset: `{DATA_PATH}`")

try:
    df = pd.read_csv(DATA_PATH)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

X = df.drop(columns=['class'])
y = df['class']

st.write("### Original Class Distribution:")
st.bar_chart(df['class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

st.write("### Class Distribution After SMOTE:")
st.bar_chart(pd.Series(y_train_res).value_counts().sort_index())

lr = LogisticRegression(max_iter=5000, multi_class='multinomial', random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

lr.fit(X_train_res, y_train_res)
rf.fit(X_train_res, y_train_res)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

st.subheader("Model Comparison")
col1, col2 = st.columns(2)
with col1:
    st.metric("Logistic Regression Accuracy", f"{acc_lr*100:.2f}%")
with col2:
    st.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")

final_model = "Logistic Regression" if acc_lr > acc_rf else "Random Forest"
st.success(f"**Best Model:** {final_model}")

st.subheader("Confusion Matrices")
col3, col4 = st.columns(2)

with col3:
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fig1, ax1 = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=ax1)
    ax1.set_title("Logistic Regression")
    st.pyplot(fig1, use_container_width=False)

with col4:
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=ax2)
    ax2.set_title("Random Forest")
    st.pyplot(fig2, use_container_width=False)

st.subheader("Classification Reports")
col5, col6 = st.columns(2)

with col5:
    st.markdown("**Logistic Regression Report**")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_lr, output_dict=True)).transpose())

with col6:
    st.markdown("**Random Forest Report**")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict=True)).transpose())


