from sklearn.pipeline import Pipeline


def build_pipeline(custom_transforms: list, estimator) -> Pipeline:
    steps = [(f"custom_{i}", t) for i, t in enumerate(custom_transforms)]
    steps.append(("model", estimator))
    return Pipeline(steps)
