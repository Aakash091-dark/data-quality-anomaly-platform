from ml.explain.shap_explainer import get_shap_values

def generate_explanation(model, features: dict, top_k: int = 5):
    shap_values = get_shap_values(model, features)

    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    explanation = [
        {
            "feature": f,
            "impact": float(v)
        }
        for f, v in sorted_features[:top_k]
    ]

    return explanation
