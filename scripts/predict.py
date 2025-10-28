from __future__ import annotations
import argparse
import joblib
from pathlib import Path
from sklearn.datasets import load_iris


def main():
    parser = argparse.ArgumentParser(description="Fai una previsione con il modello Iris addestrato.")
    parser.add_argument(
        "--inputs",
        nargs=4,
        type=float,
        required=True,
        help="Valori numerici: sepal_length sepal_width petal_length petal_width",
    )
    parser.add_argument(
        "--model",
        default="models/model.joblib",
        help="Percorso del modello salvato (default: models/model.joblib)",
    )
    args = parser.parse_args()


    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}.")

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    iris = load_iris()
    class_names = iris.target_names

    # Predici
    prediction = pipeline.predict([args.inputs])[0]
    pred_label = class_names[prediction]

    print(f"Input: {args.inputs}")
    print(f"Predicted class index: {prediction}")
    print(f"Predicted class: {pred_label}")


if __name__ == "__main__":
    main()
