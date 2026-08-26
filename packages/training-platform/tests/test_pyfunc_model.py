import numpy as np
import pandas as pd

from training_platform.pyfunc_model import FeaturePlatformModel


class _FakeSklearnModel:
    feature_names_in_ = ["txn_count", "avg_ticket"]

    def predict_proba(self, X):
        assert list(X.columns) == list(self.feature_names_in_)
        return np.array([[0.9, 0.1], [0.2, 0.8]])


def test_predict_returns_positive_class_probability():
    wrapped = FeaturePlatformModel(_FakeSklearnModel())
    model_input = pd.DataFrame({"txn_count": [3, 5], "avg_ticket": [10.0, 20.0]})
    result = wrapped.predict(context=None, model_input=model_input)
    assert list(result) == [0.1, 0.8]


def test_predict_filters_and_reorders_columns_unseen_at_fit_time():
    wrapped = FeaturePlatformModel(_FakeSklearnModel())
    model_input = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "reference_date": ["2026-08-24", "2026-08-24"],
            "avg_ticket": [10.0, 20.0],
            "txn_count": [3, 5],
        }
    )
    result = wrapped.predict(context=None, model_input=model_input)
    assert list(result) == [0.1, 0.8]
