from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier

from training_platform.pipeline import build_pipeline


class _AddOneTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + 1


def test_build_pipeline_puts_estimator_last():
    pipeline = build_pipeline([_AddOneTransform()], DummyClassifier(strategy="constant", constant=0))
    step_names = [name for name, _ in pipeline.steps]
    assert step_names[-1] == "model"
    assert step_names[0] == "custom_0"


def test_build_pipeline_with_no_custom_transforms():
    pipeline = build_pipeline([], DummyClassifier(strategy="constant", constant=0))
    assert [name for name, _ in pipeline.steps] == ["model"]


def test_build_pipeline_is_fittable_and_predictable():
    import pandas as pd

    pipeline = build_pipeline([_AddOneTransform()], DummyClassifier(strategy="constant", constant=1))
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 1, 1])
    pipeline.fit(X, y)
    assert list(pipeline.predict(X)) == [1, 1, 1]
