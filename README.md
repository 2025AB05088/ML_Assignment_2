# Machine Learning Classification Assignment

## Problem Statement
The goal of this assignment is to implement multiple machine learning classification models to predict the diagnosis of breast cancer (malignant or benign) based on various features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. We aim to compare the performance of different algorithms and deploy the best solution as an interactive web application.

## Dataset Description
**Dataset Name:** 
**Source:** 
**Characteristics:**
- **Instances:** 
- **Features:** 
- **Classes:** 
- **Missing Values:**

The features describe characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Models Implemented
I tested the following classifiers to see which performed best:
1.  **Logistic Regression**: A solid baseline for binary classification.
2.  **Decision Tree**: Easy to interpret but prone to overfitting.
3.  **K-Nearest Neighbors (KNN)**: Classifies based on similarity to nearby points.
4.  **Naive Bayes (Gaussian)**: Fast and effective, assuming feature independence.
5.  **Random Forest**: An ensemble of trees to improve accuracy and reduce overfitting.
6.  **XGBoost**: A powerful gradient boosting algorithm known for high performance.

## Performance Comparison
Here are the results from my testing on a 20% hold-out set:

| Model               |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression |     000000 | 000000 |      000000 |   000000 |     000000 | 000000 |
| Decision Tree       |     000000 | 000000 |      000000 |   000000 |     000000 | 000000 |
| KNN                 |     000000 | 000000 |      000000 |   000000 |     000000 | 000000 |
| Naive Bayes         |     000000 | 000000 |      000000 |   000000 |     000000 | 000000 |
| Random Forest       |     000000 | 000000 |      000000 |   000000 |     000000 | 000000 |
| XGBoost             |     000000 | 000000 |      000000 |   000000 |     000000 | 000000 |

## 5. Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** |  |
| **Decision Tree** |  |
| **KNN** |  |
| **Naive Bayes** |  |
| **Random Forest** |  |
| **XGBoost** |  |

## How to Run the App
I've set up the app to train on a local file and test on uploads.

1.  **Setup**: Make sure `train.csv` is in the folder. (You can run `generate_data.py` if you need a fresh copy).
2.  **Install**: `pip install -r requirements.txt`
3.  **Launch**: `streamlit run streamlit_app.py`
4.  **Use**:
    - The app trains all models automatically when it starts.
    - Upload your `test.csv` in the sidebar.
    - Pick a model to see how it performed (metrics + confusion matrix).
