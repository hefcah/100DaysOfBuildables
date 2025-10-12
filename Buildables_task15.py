# Buildables_task15.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    df = pd.read_csv(r"C:\\Users\\ma007\\OneDrive\\Desktop\\Buildables_15\\data.csv")

    # Drop unnamed or empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Encode target column
    if 'diagnosis' in df.columns:
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M=1, B=0
    else:
        raise ValueError("Target column 'diagnosis' not found in dataset.")

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Fill missing values if any
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def train_model():
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # Train models
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Predictions
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # Evaluation
    results = {
        "Decision Tree": {
            "Accuracy": accuracy_score(y_test, dt_pred),
            "Precision": precision_score(y_test, dt_pred),
            "Recall": recall_score(y_test, dt_pred)
        },
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred),
            "Recall": recall_score(y_test, rf_pred)
        },
        "XGBoost": {
            "Accuracy": accuracy_score(y_test, xgb_pred),
            "Precision": precision_score(y_test, xgb_pred),
            "Recall": recall_score(y_test, xgb_pred)
        }
    }

    # Display model comparison
    print(" Model Performance Comparison:")
    for model, metrics in results.items():
        print(f"{model}: Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}")

    # Feature importance (for Random Forest)
    feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_imp[:10], y=feature_imp.index[:10])
    plt.title("Top 10 Important Features - Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    # Return best model (Random Forest)
    return rf

# If run directly
if __name__ == "__main__":
    model = train_model()
    print("\n Model trained and ready for Streamlit app.")
