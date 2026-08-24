import mlflow.pyfunc


class FeaturePlatformModel(mlflow.pyfunc.PythonModel):
    """Default: retorna P(classe positiva) como coluna double, igual ao
    ProbabilityScorer validado na POC. Sobrescreva predict() para outro comportamento."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)[:, 1]
