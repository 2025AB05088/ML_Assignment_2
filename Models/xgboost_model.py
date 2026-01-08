import xgboost as xgb

def get_model(random_state=42):
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
