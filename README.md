# Steel Plates Fault Classification

## Problem Statement
For this assignment, I worked on predicting different types of faults in steel plates using machine learning. The dataset has 7 different fault types, so it's a multi-class classification problem. I trained 6 different models, tuned their hyperparameters, and built a Streamlit app to showcase the results.

## Dataset Description
**Dataset Name:** Steel Plates Faults Dataset  
**Source:** UCI Machine Learning Repository  
**Characteristics:**
- **Training Instances:** 1,552
- **Features:** 27 numeric attributes
- **Classes:** 7 types of steel plate faults (multi-class classification)
- **Missing Values:** None

The features describe things like geometric properties, luminosity values, and shape characteristics of the defects.

## Models Used

I implemented 6 different classifiers with feature scaling and RandomizedSearchCV for hyperparameter optimization:

1. **Logistic Regression** - Linear model with L2 regularization (max_iter=20,000, C=42.99, solver=lbfgs)
2. **Decision Tree** - Entropy-based classifier (max_depth=19, min_samples_split=11, min_samples_leaf=5)
3. **K-Nearest Neighbors** - Distance-weighted learning with feature scaling (k=6, ball_tree algorithm, Minkowski p=3)
4. **Naive Bayes** - Gaussian probabilistic classifier with smoothing (var_smoothing=7.70e-06)
5. **Random Forest** - Ensemble of 210 decision trees (max_depth=30, no bootstrap, balanced_subsample weighting)
6. **XGBoost** - Gradient boosting (159 estimators, max_depth=8, learning_rate=0.27, colsample=0.99)

## Results on Test Set

Here's how each model performed on the test data:

| Model               |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:--------------------|-----------|--------|-------------|----------|------------|--------|
| Logistic Regression |     0.7147 | 0.8957 |      0.7137 |   0.7147 |     0.7125 | 0.6371 |
| Decision Tree       |     0.7378 | 0.8839 |      0.7461 |   0.7378 |     0.7403 | 0.6666 |
| KNN                 |     0.7481 | 0.8951 |      0.7474 |   0.7481 |     0.7453 | 0.6790 |
| Naive Bayes         |     0.6375 | 0.8887 |      0.7112 |   0.6375 |     0.6282 | 0.5806 |
| Random Forest       |     0.8252 | 0.9469 |      0.8280 |   0.8252 |     0.8258 | 0.7763 |
| XGBoost             |     0.7841 | 0.9469 |      0.7840 |   0.7841 |     0.7836 | 0.7220 |

## Observations

**Random Forest** achieved the best performance with 82.52% accuracy. The hyperparameter search found that 210 trees with max_depth=30 and balanced_subsample class weighting (without bootstrap) was optimal for this dataset. This ensemble approach effectively handles the 7-class problem with superior generalization.

**XGBoost** came in second with 78.41% accuracy and excellent AUC at 0.9469. The key was tuning parameters like colsample_bytree (0.995), learning_rate (0.274), and 159 estimators with max_depth=8. The regularization terms (reg_alpha=0.305, reg_lambda=1.417) prevent overfitting while maintaining strong performance.

**KNN** performed well at 74.81% accuracy. The optimal configuration was k=6 neighbors with distance weighting, ball_tree algorithm, and Minkowski metric (p=3). Feature scaling with StandardScaler was essential for this distance-based method to work properly.

**Decision Tree** reached 73.78% accuracy with a moderately deep tree (max_depth=19), min_samples_split=11, and min_samples_leaf=5. The entropy criterion outperformed gini for splitting decisions in this multi-class scenario.

**Logistic Regression** achieved 71.47% accuracy. The model benefited from feature scaling and a moderate regularization parameter (C=42.99). The lbfgs solver with max_iter=20,000 worked best for this multi-class problem.

**Naive Bayes** was the weakest at 63.75% accuracy, though it still has decent AUC (0.8887) for probability estimates. The Gaussian assumption with var_smoothing=7.70e-06 doesn't fully capture the complexity in this dataset's feature relationships.

Feature scaling proved critical for distance-based and gradient-based algorithms. StandardScaler preprocessing was applied to Logistic Regression, KNN, Naive Bayes, and XGBoost through sklearn pipelines. All models were trained using RandomizedSearchCV with 500 iterations and random_state=99 for reproducibility.

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run streamlit_app.py
```

Then just upload the test CSV file and you can see predictions from any of the models. The app also shows a comparison table across all models.

## Files

```
├── Models/                    # Saved .pkl model files
├── streamlit_app.py          # Web app
├── requirements.txt          
├── steel_faults_train.csv   
├── steel_faults_test.csv    
└── readme.md                
```

---

Random Forest and XGBoost are clearly the winners here. For production use, I'd probably go with Random Forest since it's slightly more accurate and easier to interpret than XGBoost.
