# Iris Classifier Baseline


Un progetto end-to-end minimal per classificare l'Iris dataset.


## Comandi principali
- `make train` — addestra e salva il modello in `models/`
- `make evaluate` — calcola metriche (accuracy, precision, recall, f1, auc) e genera `reports/metrics.json` + `reports/shap/shap_summary.png`
- `make format | lint | test` — qualità del codice
- `make predict` - calcola predizione sui valori di [sepal_length sepal_width petal_length petal_width] passati dal makefile


## Configurazione
Vedi `config/config.yaml` per tipo modello e iperparametri. Puoi passare a `LogisticRegression` cambiando `model.type`.


## Output
- `models/model.joblib` — pipeline sklearn pronta per `predict`
- `reports/metrics.json` — metriche globali
- `reports/shap/shap_summary.png` — explainability
