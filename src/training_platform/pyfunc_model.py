import mlflow.pyfunc


class FeaturePlatformModel(mlflow.pyfunc.PythonModel):
    """Default: retorna P(classe positiva) como coluna double, igual ao
    ProbabilityScorer validado na POC. Sobrescreva predict() para outro comportamento."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        # fe.score_batch entrega model_input com colunas extras (chave de entidade,
        # chave de timestamp) que o modelo não viu no fit — feature_names_in_ (padrão
        # do sklearn, preenchido ao fitar com um DataFrame nomeado) filtra e reordena
        # para o schema exato de fit.
        model_input = model_input[self.model.feature_names_in_]
        return self.model.predict_proba(model_input)[:, 1]
