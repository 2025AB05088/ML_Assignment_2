from sklearn.tree import DecisionTreeClassifier

def get_model(random_state=42):
    return DecisionTreeClassifier(random_state=random_state)
