import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import os
from Models import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost_model

# Set page config
st.set_page_config(page_title="ML Assignment 2 - Classification App", layout="wide")

st.title("Machine Learning Classification Assignment")
st.markdown("""
Welcome! This dashboard allows you to evaluate different machine learning models on the ABC dataset.
The models are trained on a local training set, and you can upload your own test data to see how they perform.
""")

# --- Configuration ---
TRAIN_DATA_PATH = "train.csv"  # Path to the training dataset

# --- Sidebar ---
st.sidebar.header("Configuration")

# 1. Load Training Data
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

train_df = load_data(TRAIN_DATA_PATH)

if train_df is None:
    st.error(f"Training data not found at `{TRAIN_DATA_PATH}`. Please add the file to the directory.")
    st.stop()

st.sidebar.success(f"Loaded Training Data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

# 2. Select Target Column
target_col = st.sidebar.selectbox("Select Target Column", train_df.columns, index=len(train_df.columns)-1)

# Prepare Training Data
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# 3. Model Training (Train ALL models)
st.sidebar.subheader("Model Hyperparameters")
n_neighbors = st.sidebar.slider("KNN: Neighbors (K)", 1, 20, 5)
n_estimators_rf = st.sidebar.slider("Random Forest: Trees", 10, 200, 100)

@st.cache_resource
def train_models(X, y, k, n_est_rf):
    models = {
        "Logistic Regression": logistic_regression.get_model(),
        "Decision Tree": decision_tree.get_model(),
        "K-Nearest Neighbors": knn.get_model(n_neighbors=k),
        "Naive Bayes": naive_bayes.get_model(),
        "Random Forest": random_forest.get_model(n_estimators=n_est_rf),
        "XGBoost": xgboost_model.get_model()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    return trained_models

with st.spinner("Training models on `train.csv`..."):
    trained_models = train_models(X_train, y_train, n_neighbors, n_estimators_rf)
    st.sidebar.success("All models trained successfully!")

# --- Main Area ---

# 4. Upload Test Data
st.subheader("1. Upload Test Dataset")
test_file = st.file_uploader("Upload your Test CSV file", type=["csv"])

if test_file is not None:
    try:
        test_df = pd.read_csv(test_file)
        st.write(f"Test Data Loaded: {test_df.shape[0]} rows")
        
        if target_col not in test_df.columns:
            st.error(f"Target column `{target_col}` not found in test data. Please ensure the test data has the same structure as training data.")
        else:
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            # Ensure feature columns match
            # (Simple check: number of columns. Ideally check names)
            if X_test.shape[1] != X_train.shape[1]:
                 st.warning(f"Feature mismatch! Train has {X_train.shape[1]} features, Test has {X_test.shape[1]}. Predictions might fail.")

            # 5. Select Model for Evaluation
            st.subheader("2. Select Model to Evaluate")
            selected_model_name = st.selectbox("Choose a model", list(trained_models.keys()))
            
            if selected_model_name:
                model = trained_models[selected_model_name]
                
                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
                except:
                    auc = "N/A"

                st.write(f"### Results for {selected_model_name}")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Precision", f"{prec:.4f}")
                col3.metric("Recall", f"{rec:.4f}")
                col4.metric("F1 Score", f"{f1:.4f}")
                col5.metric("MCC", f"{mcc:.4f}")
                col6.metric("AUC", f"{auc if isinstance(auc, str) else f'{auc:.4f}'}")

                col_cm, col_cr = st.columns(2)
                
                with col_cm:
                    st.write("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                
                with col_cr:
                    st.write("#### Classification Report")
                    st.text(classification_report(y_test, y_pred))

    except Exception as e:
        st.error(f"Error processing test file: {e}")
else:
    st.info("Please upload a CSV file to evaluate the models.")

