import shap
import pandas as pd

def get_shap_values(model, features: dict):
    """
    Returns SHAP values for a single feature vector
    """
    X = pd.DataFrame([features])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return dict(zip(X.columns, shap_values[0]))
import shap
import pandas as pd

def get_shap_values(model, features: dict):
    """
    Returns SHAP values for a single feature vector
    """
    X = pd.DataFrame([features])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return dict(zip(X.columns, shap_values[0]))
