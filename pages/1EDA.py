import streamlit as st
import pandas as pd

st.title("COVID Dataset - EDA")

df = pd.read_csv("covid_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Summary Statistics")
st.write(df.describe())

st.subheader("COVID Positive Count")
st.bar_chart(df["covid_positive"].value_counts())