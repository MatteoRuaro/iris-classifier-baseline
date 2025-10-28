from iris_pipeline.data import load_data
from iris_pipeline.model import ModelSpec, build_pipeline


def test_pipeline_fits():
    X, y, _ = load_data()
    spec = ModelSpec("RandomForestClassifier", {"n_estimators": 10, "random_state": 0})
    pipe = build_pipeline(spec)
    pipe.fit(X, y)
    assert hasattr(pipe, "predict")
