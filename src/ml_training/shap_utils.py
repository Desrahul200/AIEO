# src/ml_training/shap_utils.py
import shap, numpy as np, json, matplotlib.pyplot as plt

def compute_shap(model, X_df, output_path: str, sample_size: int = 10000):
    # sample to keep it fast & deterministic
    X = X_df.sample(min(sample_size, len(X_df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    # global mean |shap|
    abs_mean = np.abs(shap_values).mean(axis=0)
    shap_dict = {feat: float(val) for feat, val in zip(X.columns.tolist(), abs_mean)}
    shap_dict = dict(sorted(shap_dict.items(), key=lambda kv: kv[1], reverse=True))
    return shap_values, explainer, shap_dict
