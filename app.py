import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Student Result Predictor", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/student_success_dataset.csv")

@st.cache_resource
def train_models(data):
    X = data.drop(columns=["passed"])
    y = data["passed"]

    numeric_features = [
        "age",
        "attendance_percent",
        "study_hours_per_day",
        "previous_marks",
        "assignment_score",
        "sleep_hours",
    ]
    categorical_features = [
        "gender",
        "internet_access",
        "extra_classes",
        "parental_support",
    ]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", encoder, categorical_features),
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", RandomForestClassifier(n_estimators=120, random_state=42, max_depth=8)),
            ]
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, y_prob),
                "Training Time Seconds": training_time,
            }
        )
        trained_models[name] = model
        predictions[name] = y_pred

    return trained_models, pd.DataFrame(results), X_test, y_test, predictions


def show_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix, {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)


data = load_data()
models, results_df, X_test, y_test, predictions = train_models(data)

st.title("Student Exam Result Prediction System")
st.write(
    "This simple machine learning web app predicts whether a student is likely to pass or fail. "
    "It also compares two machine learning models on the same dataset."
)

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Dashboard", "Dataset"])

with tab1:
    st.header("Enter Student Data")
    selected_model = st.selectbox("Choose model for prediction", list(models.keys()))

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 16, 25, 20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        attendance_percent = st.slider("Attendance Percentage", 40.0, 100.0, 78.0)
        study_hours_per_day = st.slider("Study Hours Per Day", 0.0, 10.0, 4.0)
        previous_marks = st.slider("Previous Marks", 0.0, 100.0, 65.0)
    with col2:
        assignment_score = st.slider("Assignment Score", 0.0, 100.0, 70.0)
        sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        extra_classes = st.selectbox("Extra Classes", ["No", "Yes"])
        parental_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])

    user_input = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "attendance_percent": attendance_percent,
                "study_hours_per_day": study_hours_per_day,
                "previous_marks": previous_marks,
                "assignment_score": assignment_score,
                "sleep_hours": sleep_hours,
                "internet_access": internet_access,
                "extra_classes": extra_classes,
                "parental_support": parental_support,
            }
        ]
    )

    if st.button("Predict Result"):
        model = models[selected_model]
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]

        if prediction == 1:
            st.success(f"Prediction: Pass. Pass probability: {probability:.2%}")
        else:
            st.error(f"Prediction: Fail. Pass probability: {probability:.2%}")

        st.write("Input used for prediction")
        st.dataframe(user_input)

with tab2:
    st.header("Model Comparison Dashboard")
    st.write("Both models are trained and tested on the same dataset.")

    display_results = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        display_results[col] = (display_results[col] * 100).round(2)
    display_results["Training Time Seconds"] = display_results["Training Time Seconds"].round(4)
    st.dataframe(display_results, use_container_width=True)

    st.subheader("Performance Chart")
    chart_data = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
    st.bar_chart(chart_data)

    st.subheader("Training Time Chart")
    time_data = results_df.set_index("Model")[["Training Time Seconds"]]
    st.bar_chart(time_data)

    selected_cm_model = st.selectbox("Choose model for confusion matrix", list(models.keys()), key="cm_model")
    show_confusion_matrix(y_test, predictions[selected_cm_model], selected_cm_model)

with tab3:
    st.header("Dataset Preview")
    st.write(f"Total records: {len(data)}")
    st.write(f"Total columns: {len(data.columns)}")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Target Count")
    target_count = data["passed"].map({0: "Fail", 1: "Pass"}).value_counts()
    st.bar_chart(target_count)
