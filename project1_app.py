import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.title(" House Price Analysis & Prediction App")

df = pd.read_csv("C:\\Users\\ma007\\OneDrive\\Desktop\\app\\Housing.csv")   
st.subheader(" Dataset Preview")
st.dataframe(df.head())

#  Clean Missing Values

st.subheader(" Data Cleaning")
df = df.dropna()
st.write("Missing values handled (dropped rows with NaN).")

# Encoding of categorical variables
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 3. Summary Stats

st.subheader(" Summary Statistics")
st.write(df.describe())

st.sidebar.header(" Filters")
price_filter = st.sidebar.slider("Select Maximum Price", 
                                 int(df['price'].min()), 
                                 int(df['price'].max()), 
                                 int(df['price'].max()))

filtered_df = df[df['price'] <= price_filter]
st.subheader("💡 Filtered Dataset")
st.dataframe(filtered_df.head())

# EDA Visualizations

st.subheader("Data Visualizations")

# Histogram
st.write("Histogram of House Prices")
fig, ax = plt.subplots()
sns.histplot(df['price'], kde=True, ax=ax)
st.pyplot(fig)

# Scatterplot
st.write("Scatterplot: Price vs Area")
fig, ax = plt.subplots()
sns.scatterplot(x=df['area'], y=df['price'], ax=ax)
st.pyplot(fig)

# Bar chart
st.write("Bar Chart: Furnishing Status")
fig, ax = plt.subplots()
sns.countplot(x=df['furnishingstatus'], ax=ax)
st.pyplot(fig)

# Modeling - Regression

st.subheader("Modeling - Regression (Predict Price)")

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# 7. Predictions

st.subheader(" Make Your Own Prediction")

# Input sliders for features (use first 5 for demo)
inputs = {}
for col in X.columns[:5]:
    val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    inputs[col] = val

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Fill missing columns with mean
for col in X.columns:
    if col not in input_df:
        input_df[col] = df[col].mean()

prediction = model.predict(input_df)[0]
st.write(f" **Predicted House Price:** {prediction:.2f}")
