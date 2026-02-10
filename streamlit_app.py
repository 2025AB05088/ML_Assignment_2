import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import os
import plotly.express as px
import plotly.graph_objects as go
from Models import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost_model

# Set page config
st.set_page_config(page_title="Machine Learning Assignment 2 - Classification Models Comparison", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.5em;
        margin-bottom: 25px;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Steel Plates Fault Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Models Comparison Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>Welcome!</strong> This dashboard allows you to evaluate different machine learning models on the <strong>Steel Plates Faults</strong> dataset.
    The models are pre-trained with optimized hyperparameters. Upload your test dataset to see predictions and performance metrics.
</div>
""", unsafe_allow_html=True)

# --- Configuration ---
TRAIN_DATA_PATH = "steel_faults_train.csv"  # Path to the training dataset

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
def load_models():
    """Load pre-trained models from .pkl files"""
    import pickle
    
    model_files = {
        "Logistic Regression": "Models/logistic_regression_trained.pkl",
        "Decision Tree": "Models/decision_tree_trained.pkl",
        "K-Nearest Neighbors": "Models/knn_trained.pkl",
        "Naive Bayes": "Models/naive_bayes_trained.pkl",
        "Random Forest": "Models/random_forest_trained.pkl",
        "XGBoost": "Models/xgboost_trained.pkl"
    }
    
    loaded_models = {}
    for name, pkl_path in model_files.items():
        if not os.path.exists(pkl_path):
            st.error(f"Model file not found: {pkl_path}. Please run train_models.py first.")
            st.stop()
        
        with open(pkl_path, 'rb') as f:
            loaded_models[name] = pickle.load(f)
    
    return loaded_models

with st.spinner("Loading pre-trained models..."):
    trained_models = load_models()
    st.success("All pre-trained models loaded successfully!")

# --- Main Area ---

# Upload Test Data
st.markdown('<h2 class="section-header">Step 1: Upload Test Dataset</h2>', unsafe_allow_html=True)

# Download button for test dataset
@st.cache_data
def fetch_test_csv():
    import requests
    url = "https://raw.githubusercontent.com/2025AB05088/ML_Assignment_2/main/steel_faults_test.csv"
    response = requests.get(url)
    response.raise_for_status()
    return response.content

try:
    test_csv_data = fetch_test_csv()
    st.markdown("You can download the test dataset here:")
    st.download_button(
        label="Download steel_faults_test.csv",
        data=test_csv_data,
        file_name="steel_faults_test.csv",
        mime="text/csv"
    )
except Exception:
    st.info("Could not fetch the test dataset from GitHub. Please download it manually from: https://github.com/2025AB05088/ML_Assignment_2/blob/main/steel_faults_test.csv")

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
            if X_test.shape[1] != X_train.shape[1]:
                 st.warning(f"Feature mismatch! Train has {X_train.shape[1]} features, Test has {X_test.shape[1]}. Predictions might fail.")

            # Select Model for Evaluation
            st.markdown('<h2 class="section-header">Step 2: Select Model to Evaluate</h2>', unsafe_allow_html=True)
            selected_model_name = st.selectbox("Choose a model", list(trained_models.keys()))
            
            if selected_model_name:
                model = trained_models[selected_model_name]
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # AUC for multi-class
                try:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)
                        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                    else:
                        auc = 0.0
                except:
                    auc = 0.0

                st.markdown(f'<h2 style="color: #2c3e50; border-left: 5px solid #3498db; padding-left: 15px; margin-top: 30px;">Results for {selected_model_name}</h2>', unsafe_allow_html=True)
                st.markdown("---")

                # Metrics Row
                st.markdown('<h3 style="color: #16a085; margin-bottom: 20px;">Key Performance Indicators</h3>', unsafe_allow_html=True)
                cols = st.columns(6)
                metrics = [
                    ("Accuracy", acc), ("Precision", prec), ("Recall", rec),
                    ("F1 Score", f1), ("MCC", mcc), ("AUC", auc)
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    val_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    col.metric(label, val_str)
                
                st.markdown("---")

                # Visualizations
                st.markdown('<h3 style="color: #c0392b; margin-top: 30px; margin-bottom: 20px;">Confusion Matrix</h3>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, 
                           linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Count'})
                ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', color='#2c3e50')
                ax.set_ylabel('True Label', fontsize=13, fontweight='bold', color='#2c3e50')
                ax.set_title(f'{selected_model_name} - Confusion Matrix', fontsize=15, fontweight='bold', pad=20, color='#34495e')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown('<h3 style="color: #8e44ad; margin-top: 30px; margin-bottom: 20px;">Model Comparison Across All Algorithms</h3>', unsafe_allow_html=True)
                
                comparison_results = []
                for name, clf in trained_models.items():
                    # Predict
                    y_p = clf.predict(X_test)
                    
                    # Metrics
                    a = accuracy_score(y_test, y_p)
                    p = precision_score(y_test, y_p, average='weighted', zero_division=0)
                    r = recall_score(y_test, y_p, average='weighted', zero_division=0)
                    f = f1_score(y_test, y_p, average='weighted', zero_division=0)
                    m = matthews_corrcoef(y_test, y_p)
                    
                    # AUC for multi-class
                    try:
                        if hasattr(clf, "predict_proba"):
                            y_pb = clf.predict_proba(X_test)
                            au = roc_auc_score(y_test, y_pb, multi_class='ovr', average='weighted')
                        else:
                            au = 0.0
                    except:
                        au = 0.0
                    
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
                
                # Interactive visualizations
                st.markdown("---")
                st.markdown('<h3 style="color: #27ae60; margin-top: 30px; margin-bottom: 20px;">Performance Metrics Visualization</h3>', unsafe_allow_html=True)
                
                # Create tabs for different metrics
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Accuracy", "Precision", "Recall", "F1 Score", "MCC", "AUC"])
                
                # Color palette for consistent styling
                colors = px.colors.qualitative.Set2
                
                with tab1:
                    fig_acc = px.bar(comparison_df, x='Model', y='Accuracy', 
                                    title='Model Comparison - Accuracy',
                                    color='Model', color_discrete_sequence=colors,
                                    text='Accuracy')
                    fig_acc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_acc.update_layout(showlegend=False, yaxis_range=[0, 1], 
                                         xaxis_title="Model", yaxis_title="Accuracy Score",
                                         height=500)
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with tab2:
                    fig_prec = px.bar(comparison_df, x='Model', y='Precision', 
                                     title='Model Comparison - Precision',
                                     color='Model', color_discrete_sequence=colors,
                                     text='Precision')
                    fig_prec.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_prec.update_layout(showlegend=False, yaxis_range=[0, 1],
                                          xaxis_title="Model", yaxis_title="Precision Score",
                                          height=500)
                    st.plotly_chart(fig_prec, use_container_width=True)
                
                with tab3:
                    fig_rec = px.bar(comparison_df, x='Model', y='Recall', 
                                    title='Model Comparison - Recall',
                                    color='Model', color_discrete_sequence=colors,
                                    text='Recall')
                    fig_rec.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_rec.update_layout(showlegend=False, yaxis_range=[0, 1],
                                         xaxis_title="Model", yaxis_title="Recall Score",
                                         height=500)
                    st.plotly_chart(fig_rec, use_container_width=True)
                
                with tab4:
                    fig_f1 = px.bar(comparison_df, x='Model', y='F1 Score', 
                                   title='Model Comparison - F1 Score',
                                   color='Model', color_discrete_sequence=colors,
                                   text='F1 Score')
                    fig_f1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_f1.update_layout(showlegend=False, yaxis_range=[0, 1],
                                        xaxis_title="Model", yaxis_title="F1 Score",
                                        height=500)
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                with tab5:
                    fig_mcc = px.bar(comparison_df, x='Model', y='MCC', 
                                    title="Model Comparison - Matthew's Correlation Coefficient",
                                    color='Model', color_discrete_sequence=colors,
                                    text='MCC')
                    fig_mcc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_mcc.update_layout(showlegend=False, yaxis_range=[-1, 1],
                                         xaxis_title="Model", yaxis_title="MCC",
                                         height=500)
                    st.plotly_chart(fig_mcc, use_container_width=True)
                
                with tab6:
                    fig_auc = px.bar(comparison_df, x='Model', y='AUC', 
                                    title='Model Comparison - AUC (Area Under Curve)',
                                    color='Model', color_discrete_sequence=colors,
                                    text='AUC')
                    fig_auc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_auc.update_layout(showlegend=False, yaxis_range=[0, 1],
                                         xaxis_title="Model", yaxis_title="AUC Score",
                                         height=500)
                    st.plotly_chart(fig_auc, use_container_width=True)



    except Exception as e:
        st.error(f"Error processing test file: {e}")
else:
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 20px; border-left: 5px solid #ffc107; border-radius: 5px; margin-top: 20px;'>
        <p style='margin: 0; color: #856404; font-size: 16px;'><strong>Info:</strong> Please upload a CSV file to evaluate the models.</p>
    </div>
    """, unsafe_allow_html=True)

