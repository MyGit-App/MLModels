import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Confusion Matrix – COVID Predictor")

# Load dataset
df = pd.read_csv("covid_data.csv")
X = df.drop(columns=["covid_positive"])
y = df["covid_positive"]

# Load trained model
try:
    model = joblib.load("logistic_model.joblib")
except:
    st.error("Model not found. Please train the Logistic Regression model first.")
    st.stop()

# Predict
y_pred = model.predict(X)

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

st.subheader("Confusion Matrix (Numbers)")
st.write(cm)

# Heatmap
st.subheader("Confusion Matrix Heatmap")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown("""
**Interpretation:**
- **True Positive (TP):** Model predicted COVID positive correctly  
- **True Negative (TN):** Model predicted COVID negative correctly  
- **False Positive (FP):** Model predicted positive but was wrong  
- **False Negative (FN):** Model predicted negative but was wrong  
""")