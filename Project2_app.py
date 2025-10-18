import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#  Streamlit UI
st.title(" Student Exam Performance Predictor")

st.write("Upload your dataset to predict student exam performance based on study, sleep, attendance, and previous scores.")

#  File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Reading the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # Feature and target selection
    X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
    y = df['exam_score']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training (Regressor)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader(" Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Prediction input
    st.subheader(" Predict Exam Score for a New Student")

    hours_studied = st.number_input("Hours Studied", min_value=0.0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
    attendance_percent = st.number_input("Attendance Percent", min_value=0.0, max_value=100.0)
    previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0)

    if st.button("Predict Exam Score"):
        new_data = pd.DataFrame([[hours_studied, sleep_hours, attendance_percent, previous_scores]],
                                columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'])
        prediction = model.predict(new_data)
        st.success(f" Predicted Exam Score: {prediction[0]:.2f}")
else:
    st.warning("Please upload a CSV file to continue.")
