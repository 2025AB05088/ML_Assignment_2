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

1. **Logistic Regression** - Linear model with L2 regularization (max_iter=100,000)
2. **Decision Tree** - Tree-based classifier with entropy criterion
3. **K-Nearest Neighbors** - Distance-based learning with feature scaling
4. **Naive Bayes** - Gaussian probabilistic classifier
5. **Random Forest** - Ensemble of 458 decision trees without bootstrap
6. **XGBoost** - Gradient boosting with extensive regularization tuning

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

**Random Forest** achieved the best performance with 80.72% accuracy. The hyperparameter search found that 458 trees without bootstrap sampling was optimal for this dataset. This ensemble approach effectively handles the 7-class problem.

**XGBoost** came in second with 78.66% accuracy and the highest AUC at 0.9447. The key was tuning parameters like colsample_bytree (0.808), gamma (0.147), and the regularization terms (reg_alpha, reg_lambda). These prevent overfitting while maintaining good performance.

**KNN** performed well at 72.24% accuracy. The optimal configuration was k=3 neighbors with distance weighting and the ball_tree algorithm. Feature scaling with StandardScaler was essential for this distance-based method to work properly.

**Logistic Regression** reached 71.47% accuracy. The model benefited significantly from feature scaling and a higher regularization parameter (C=73.2). The lbfgs solver worked best for this multi-class problem.

**Decision Tree** also achieved 71.47% accuracy with a deeper tree (max_depth=45) and larger leaf nodes (min_samples_leaf=9). The entropy criterion outperformed gini for splitting decisions.

**Naive Bayes** was the weakest at 63.75% accuracy, though it still has decent AUC (0.8887) for probability estimates. The Gaussian assumption doesn't capture all the complexity in this dataset's feature relationships.

Feature scaling proved critical for distance-based and gradient-based algorithms. StandardScaler preprocessing was applied to Logistic Regression, KNN, Naive Bayes, and XGBoost through sklearn pipelines.

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
