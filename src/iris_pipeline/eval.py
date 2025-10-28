from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def evaluate(pipe, X, y, report_dir: str, random_state: int = 42) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_te)
    except Exception:
        pass

    acc = float(accuracy_score(y_te, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "precision_w": float(prec),
        "recall_w": float(rec),
        "f1_w": float(f1),
    }
    if y_proba is not None and y_proba.shape[1] == 3:
        # AUC macro one-vs-rest
        y_onehot = np.eye(3)[y_te]
        auc = roc_auc_score(y_onehot, y_proba, average="macro", multi_class="ovr")
        metrics["auc_macro_ovr"] = float(auc)

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
