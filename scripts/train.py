from __future__ import annotations
import argparse
import yaml
from pathlib import Path
from iris_pipeline.data import load_data
from iris_pipeline.model import ModelSpec, build_pipeline, train_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    X, y, _ = load_data()

    spec = ModelSpec(type=cfg["model"]["type"], params=cfg["model"]["params"])
    pipe = build_pipeline(spec)
    path = train_and_save(pipe, X, y, cfg["paths"]["model_dir"])
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
