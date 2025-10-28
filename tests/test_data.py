from iris_pipeline.data import load_data


def test_load_data_shapes():
    X, y, fn = load_data()
    assert X.shape[0] == y.shape[0]
    assert len(fn) == X.shape[1]
