import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.title("Train COVID Prediction Models")

# Load dataset
df = pd.read_csv("covid_data.csv")
X = df.drop(columns=["covid_positive"])
y = df["covid_positive"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("### Dataset Split")
st.write(f"Training samples: {len(X_train)}")
st.write(f"Testing samples: {len(X_test)}")

# Logistic Regression Training
if st.button("Train Logistic Regression"):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "logistic_model.joblib")

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success("Logistic Regression model trained!")
    st.write(f"**Test Accuracy:** {acc:.3f}")

# Linear Regression Training
if st.button("Train Linear Regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "linear_model.joblib")

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success("Linear Regression model trained!")
    st.write(f"**Test MSE:** {mse:.3f}")
    st.write(f"**R² Score:** {r2:.3f}")
