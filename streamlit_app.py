import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import os
from Models import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost_model

# Set page config
st.set_page_config(page_title="Machine Learning Assignment 2 - Classification Models Comparison", layout="wide")

st.title("Machine Learning Assignment 2 - Classification Models Comparison")
st.markdown("""
Welcome! This dashboard allows you to evaluate different machine learning models on the ABC dataset.
The models are trained on a local training set, and you can upload your own test data to see how they perform.
""")

# --- Configuration ---
TRAIN_DATA_PATH = "train.csv"  # Path to the training dataset

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

st.success(f"Loaded Training Data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

# Prepare Training Data
target_col = train_df.columns[-1]
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

@st.cache_resource
def train_models(X, y):
    models = {
        "Logistic Regression": logistic_regression.get_model(),
        "Decision Tree": decision_tree.get_model(),
        "K-Nearest Neighbors": knn.get_model(n_neighbors=7),
        "Naive Bayes": naive_bayes.get_model(),
        "Random Forest": random_forest.get_model(n_estimators=100),
        "XGBoost": xgboost_model.get_model()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    return trained_models

with st.spinner("Training models on `train.csv`..."):
    trained_models = train_models(X_train, y_train)
    st.success("All models trained successfully!")

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
            target_col = test_df.columns[-1]
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

                st.markdown(f"### Results for **{selected_model_name}**")
                st.markdown("---")

                # Metrics Row
                st.markdown("#### Key Performance Indicators")
                cols = st.columns(6)
                metrics = [
                    ("Accuracy", acc), ("Precision", prec), ("Recall", rec),
                    ("F1 Score", f1), ("MCC", mcc), ("AUC", auc)
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    val_str = f"{value:.4f}" if isinstance(value, (int, float)) else value
                    col.metric(label, val_str)
                
                st.markdown("---")

                # Visualizations
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)
                
                st.markdown("---")
                
                st.markdown("#### Model Comparison")
                
                comparison_results = []
                for name, clf in trained_models.items():
                    # Predict
                    y_p = clf.predict(X_test)
                    y_pb = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
                    
                    # Metrics
                    a = accuracy_score(y_test, y_p)
                    p = precision_score(y_test, y_p, average='weighted', zero_division=0)
                    r = recall_score(y_test, y_p, average='weighted', zero_division=0)
                    f = f1_score(y_test, y_p, average='weighted', zero_division=0)
                    m = matthews_corrcoef(y_test, y_p)
                    try:
                        au = roc_auc_score(y_test, y_pb) if y_pb is not None else 0
                    except:
                        au = 0
                    
                    comparison_results.append({
                        "Model": name,
                        "Accuracy": a,
                        "Precision": p,
                        "Recall": r,
                        "F1 Score": f,
                        "MCC": m,
                        "AUC": au
                    })
                
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "MCC": "{:.4f}",
                    "AUC": "{:.4f}"
                }), use_container_width=True)



    except Exception as e:
        st.error(f"Error processing test file: {e}")
else:
    st.info("Please upload a CSV file to evaluate the models.")

