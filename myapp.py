import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# App title

st.title("Breast Cancer Prediction App")

st.sidebar.header("Model Settings")

# Let user upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Preprocessing
    
    # Drop columns that are not useful
    if "id" in data.columns:
        data = data.drop(columns=["id"])
    if "Unnamed: 32" in data.columns:
        data = data.drop(columns=["Unnamed: 32"])

    # Encode target variable (diagnosis: M = 1, B = 0)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling (important for Logistic & KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Sidebar Model Selection
    # -----------------------------
    model_choice = st.sidebar.selectbox(
        "Choose a Model",
        ("Logistic Regression", "Decision Tree", "KNN")
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = KNeighborsClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Results
   
    st.subheader(f"Results using {model_choice}")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV file to proceed.")
