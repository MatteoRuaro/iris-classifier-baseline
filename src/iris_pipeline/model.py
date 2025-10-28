from __future__ import annotations
import os
import joblib
from dataclasses import dataclass
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelSpec:
    type: str
    params: dict[str, Any]


def build_estimator(spec: ModelSpec):
    if spec.type == "RandomForestClassifier":
        return RandomForestClassifier(**spec.params)
    if spec.type == "LogisticRegression":
        return LogisticRegression(max_iter=1000, **spec.params)
    raise ValueError(f"Modello non supportato: {spec.type}")


def build_pipeline(spec: ModelSpec) -> Pipeline:
    est = build_estimator(spec)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", est),
        ]
    )
    return pipe


def train_and_save(pipe: Pipeline, X, y, model_dir: str) -> str:
    pipe.fit(X, y)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "model.joblib")
    joblib.dump({"pipeline": pipe}, path)
    return path
