import numpy as np

from training_platform.pyfunc_model import FeaturePlatformModel


class _FakeSklearnModel:
    def predict_proba(self, X):
        return np.array([[0.9, 0.1], [0.2, 0.8]])


def test_predict_returns_positive_class_probability():
    wrapped = FeaturePlatformModel(_FakeSklearnModel())
    result = wrapped.predict(context=None, model_input=None)
    assert list(result) == [0.1, 0.8]
