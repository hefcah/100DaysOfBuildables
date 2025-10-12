# app_15.py
import streamlit as st
import numpy as np
from Buildables_task15 import train_model

@st.cache_resource
def get_model():
    model = train_model()
    return model

model = get_model()

st.title(" Breast Cancer Classification (Random Forest)")
st.write("Enter cell nucleus measurements to predict if the tumor is **Benign (B)** or **Malignant (M)**.")

# Input features (based on dataset columns)
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

user_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}", min_value=0.0, value=10.0)
    user_data.append(value)

if st.button(" Predict"):
    input_data = np.array(user_data).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    result = "Malignant (M)" if prediction == 1 else "Benign (B)"
    st.success(f" Prediction: The tumor is **{result}**")

st.image("feature_importance.png", caption="Top 10 Important Features", use_container_width=True)
