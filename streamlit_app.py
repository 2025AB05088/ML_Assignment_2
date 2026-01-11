import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import os
from Models import logistic_regression, decision_tree, knn, naive_bayes, random_forest, xgboost_model

# Set page config
st.set_page_config(
    page_title="Steel Plates Fault Classification", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title with styling
st.markdown("""<h1 style='text-align: center; color: #1f77b4;'>🔬 Steel Plates Fault Classification</h1>""", unsafe_allow_html=True)
st.markdown("""<h3 style='text-align: center; color: #555;'>Machine Learning Models Comparison</h3>""", unsafe_allow_html=True)
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='font-size: 16px; color: #333;'>
    👋 Welcome! This dashboard evaluates different machine learning models on the <b>Steel Plates Faults</b> dataset.
    The models are pre-trained with optimized hyperparameters. Upload <code>steel_faults_test.csv</code> to see predictions and performance metrics.
    </p>
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

st.success(f"✅ Loaded Training Data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

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
    st.success("✅ All pre-trained models loaded successfully!")

# --- Main Area ---

# 4. Upload Test Data
st.markdown("""<h2 style='color: #e377c2;'>📤 Step 1: Upload Test Dataset</h2>""", unsafe_allow_html=True)
test_file = st.file_uploader("Upload your Test CSV file", type=["csv"])

if test_file is not None:
    try:
        test_df = pd.read_csv(test_file)
        st.write(f"✅ Test Data Loaded: {test_df.shape[0]} rows")
        
        if target_col not in test_df.columns:
            st.error(f"Target column `{target_col}` not found in test data. Please ensure the test data has the same structure as training data.")
        else:
            target_col = test_df.columns[-1]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            # Ensure feature columns match
            if X_test.shape[1] != X_train.shape[1]:
                 st.warning(f"Feature mismatch! Train has {X_train.shape[1]} features, Test has {X_test.shape[1]}. Predictions might fail.")

            # 5. Select Model for Evaluation
            st.markdown("""<h2 style='color: #8c564b;'>🤖 Step 2: Select Model to Evaluate</h2>""", unsafe_allow_html=True)
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

                st.markdown(f"""<h2 style='color: #1f77b4;'>📊 Results for <span style='color: #ff7f0e;'>{selected_model_name}</span></h2>""", unsafe_allow_html=True)
                st.markdown("---")

                # Metrics Row with icons
                st.markdown("""<h3 style='color: #2ca02c;'>📈 Key Performance Indicators</h3>""", unsafe_allow_html=True)
                cols = st.columns(6)
                metrics = [
                    ("🎯 Accuracy", acc), ("✓ Precision", prec), ("↩ Recall", rec),
                    ("⚖ F1 Score", f1), ("📉 MCC", mcc), ("📊 AUC", auc)
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    val_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    col.metric(label, val_str)
                
                st.markdown("---")

                # Visualizations
                st.markdown("""<h3 style='color: #d62728;'>🔥 Confusion Matrix</h3>""", unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax, cbar=True, 
                           linewidths=1, linecolor='white')
                ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
                ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
                ax.set_title(f'{selected_model_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("""<h3 style='color: #9467bd;'>🏆 Model Comparison Across All Algorithms</h3>""", unsafe_allow_html=True)
                
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



    except Exception as e:
        st.error(f"❌ Error processing test file: {e}")
else:
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-left: 5px solid #ffc107; border-radius: 5px;'>
        <p style='margin: 0; color: #856404;'><b>ℹ️ Info:</b> Please upload a CSV file to evaluate the models.</p>
    </div>
    """, unsafe_allow_html=True)

