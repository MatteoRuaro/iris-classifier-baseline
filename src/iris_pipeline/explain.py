from __future__ import annotations
from pathlib import Path
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless


def compute_shap(pipe, X, shap_dir: str) -> str:
    Path(shap_dir).mkdir(parents=True, exist_ok=True)
    X_sample = X.iloc[: min(100, len(X))]
    model = pipe.named_steps["model"]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(
            pipe.named_steps["scaler"].transform(X_sample)
        )
    except Exception:
        explainer = shap.KernelExplainer(pipe.predict_proba, shap.kmeans(X_sample, 10))
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    plt.figure()
    try:
        shap.summary_plot(shap_values, X_sample, show=False)
    except Exception:
        shap.summary_plot(np.array(shap_values), X_sample, show=False)
    out_path = str(Path(shap_dir) / "shap_summary.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
