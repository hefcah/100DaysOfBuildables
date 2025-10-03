import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

@st.cache_data
uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Drop unnecessary columns that do not add value
    data = data.drop(["id", "Unnamed: 32"], axis=1)
    return data

data = load_data()
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Scaling numeric features so that Logistic Regression and KNN perform better
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  dataset into training and testing portions
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_models():
    models = {}

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    models["Logistic Regression"] = log_model

    # Decision Tree
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    models["Decision Tree"] = tree_model

    # K Nearest Neighbors
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    models["KNN"] = knn_model

    return models

models = train_models()

# ------------------------------------
# Streamlit user interface
# ------------------------------------
st.title("Breast Cancer Prediction App")
st.write("This app allows you to switch between different models "
         "and test predictions interactively.")

# Sidebar for selecting which model to use
st.sidebar.title("Model Selector")
model_choice = st.sidebar.radio("Choose a model:", list(models.keys()))

# Select the chosen model
model = models[model_choice]

# Display model accuracy on the test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Currently using: {model_choice}")
st.write(f"Test Accuracy: {acc:.2f}")

# ------------------------------------
# Input form for new predictions
# ------------------------------------
st.subheader("Make a Prediction")
sample_input = []

# Create number inputs for each feature
for idx, col in enumerate(X.columns):
    val = st.number_input(
        f"Enter {col}:",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    sample_input.append(val)

# Predict when button is clicked
if st.button("Predict"):
    input_scaled = scaler.transform([sample_input])
    pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Diagnosis: {pred}")

