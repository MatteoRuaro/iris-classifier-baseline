from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import joblib
from iris_pipeline.data import load_data
from iris_pipeline.model import ModelSpec, build_pipeline
from iris_pipeline.eval import evaluate
from iris_pipeline.explain import compute_shap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    X, y, feature_names = load_data()

    spec = ModelSpec(type=cfg["model"]["type"], params=cfg["model"]["params"])
    pipe = build_pipeline(spec)
    metrics = evaluate(pipe, X, y, cfg["paths"]["report_dir"], cfg["random_state"])
    shap_path = compute_shap(pipe, X, cfg["paths"]["shap_dir"])

    joblib.dump(
        {"pipeline": pipe, "feature_names": feature_names},
        Path(cfg["paths"]["model_dir"]) / "model.joblib",
    )

    print(metrics)
    print(f"SHAP summary plot saved to {shap_path}")


if __name__ == "__main__":
    main()
