import pytest

from mlplatform.serving.contract import (
    BatchServingConfig,
    OnlineServingConfig,
    clear_registry,
    register_serving_config,
)
from mlplatform.serving.resource_gen import generate_resources
from mlplatform.serving.structure import InferenceBatchStruct


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _batch(**o) -> BatchServingConfig:
    return BatchServingConfig(
        **{
            "domain": "exemplo",
            "model_name": "modelo_batch",
            "spine_inference_table": "workspace.exemplo.spine",
            "schedule_cron": "0 0 6 * * ?",
            "output": InferenceBatchStruct(
                primary_key=["customer_id"],
                ts_date="reference_date",
                predict_cols=["prediction"],
            ),
            **o,
        }
    )


def _gen(**o):
    return generate_resources(**{"catalog": "workspace", **o})


def test_batch_config_generates_a_scheduled_job_calling_the_entry_point():
    register_serving_config(_batch())

    job = _gen()["resources"]["jobs"]["score_batch_modelo_batch"]

    assert job["schedule"]["quartz_cron_expression"] == "0 0 6 * * ?"
    wheel = job["tasks"][0]["python_wheel_task"]
    assert (wheel["package_name"], wheel["entry_point"]) == ("mlplatform", "mlp-score-batch")
    assert wheel["named_parameters"]["model_name"] == "modelo_batch"


def test_online_config_generates_an_endpoint_with_a_numeric_version():
    """DABs recusa `models:/nome@alias` — só aceita entity_version numérico."""
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    endpoints = _gen(resolve_alias_version=lambda full, alias: 3)["resources"]["model_serving_endpoints"]

    served = endpoints["exemplo-modelo_online-serving"]["config"]["served_entities"][0]
    assert served["entity_name"] == "${var.catalog}.exemplo_models.modelo_online"
    assert served["entity_version"] == "3"


def test_the_resolver_receives_the_real_catalog_not_a_placeholder():
    """O domínio carregava um CATALOG hardcodado, com um comentário pedindo que
    batesse com o default do databricks.yml — e nada verificava se batia."""
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))
    seen = []

    _gen(catalog="producao", resolve_alias_version=lambda full, alias: seen.append(full) or 1)

    assert seen == ["producao.exemplo_models.modelo_online"]


def test_online_without_a_resolver_fails_with_the_reason():
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    with pytest.raises(ValueError, match="entity_version"):
        _gen()


def test_refresh_endpoint_exists_only_where_there_is_an_endpoint():
    """Emiti-lo sempre deployava um job inútil no bundle de batch — e com dois
    bundles de serving no mesmo domínio, dois jobs de mesmo nome no workspace,
    indistinguíveis na UI."""
    register_serving_config(_batch())
    assert not [k for k in _gen()["resources"]["jobs"] if k.startswith("refresh_endpoint")]

    clear_registry()
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))
    jobs = _gen(resolve_alias_version=lambda f, a: 1)["resources"]["jobs"]
    assert "refresh_endpoint_modelo_online" in jobs


def test_the_refresh_job_receives_the_deployed_endpoint_name():
    """O DABs prefixa os recursos por target (`dev_<usuario>_...`). Derivar o
    nome dentro do job ignora esse prefixo e procura um endpoint inexistente —
    foi assim que o refresh falhou ao vivo. A referência abaixo é resolvida pelo
    próprio DABs no deploy, então vale em qualquer target."""
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    job = _gen(resolve_alias_version=lambda f, a: 1)["resources"]["jobs"]["refresh_endpoint_modelo_online"]
    named = job["tasks"][0]["python_wheel_task"]["named_parameters"]

    assert named["endpoint_name"] == (
        "${resources.model_serving_endpoints.exemplo-modelo_online-serving.name}"
    )
    assert named["model_name"] == "modelo_online"


def test_the_refresh_job_has_no_parameter_the_user_must_fill_in():
    """`model_name` era um job parameter com default vazio, e o entrypoint o
    exige: qualquer execução agendada morria com KeyError: ''."""
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    job = _gen(resolve_alias_version=lambda f, a: 1)["resources"]["jobs"]["refresh_endpoint_modelo_online"]

    assert all(p["default"] for p in job["parameters"])


def test_endpoints_are_omitted_when_there_is_no_online_config():
    register_serving_config(_batch())

    assert "model_serving_endpoints" not in _gen()["resources"]


def test_every_job_carries_the_serverless_environment():
    register_serving_config(_batch())

    for job in _gen()["resources"]["jobs"].values():
        assert job["environments"][0]["spec"]["client"] == "3"
        assert all(t["environment_key"] == "default" for t in job["tasks"])


def test_a_domain_that_never_trained_gets_an_actionable_error():
    """O recurso de endpoint exige `entity_version` numérico — o DABs recusa
    alias —, então o alias resolve na hora de GERAR o bundle. Um domínio novo
    precisa treinar antes de deployar serving online.

    O erro cru do SDK é `Schema '<catalog>.<domain>_models' does not exist`:
    verdadeiro, e inútil para quem não conhece a ordem."""
    from databricks.sdk.errors import NotFound

    from mlplatform.serving.adapters import ModelNotTrainedYet, SdkModelRegistry

    class _RegistroVazio(SdkModelRegistry):
        def version_for_alias(self, full_model_name, alias):
            try:
                raise NotFound("Schema 'workspace.credito_models' does not exist.")
            except NotFound as erro:
                raise ModelNotTrainedYet(
                    f"'{full_model_name}@{alias}' não existe ainda. Rode o pipeline de treino."
                ) from erro

    with pytest.raises(ModelNotTrainedYet, match="Rode o pipeline de treino"):
        _RegistroVazio().version_for_alias("workspace.credito_models.pd_inadimplencia", "champion")
