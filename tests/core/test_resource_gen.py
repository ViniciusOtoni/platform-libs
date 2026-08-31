from pathlib import Path

import pytest

from mlplatform.core.resource_gen import ENVIRONMENT_KEY, dump_yaml, with_environment


def test_dump_yaml_writes_the_resource_dict(tmp_path: Path):
    out = tmp_path / "generated.yml"

    dump_yaml({"resources": {"jobs": {"a_job": {"name": "a_job"}}}}, str(out))

    content = out.read_text(encoding="utf-8")
    assert "resources:" in content
    assert "a_job" in content


def _job(n_tasks: int) -> dict:
    return {"name": "a_job", "tasks": [{"task_key": f"t{i}"} for i in range(n_tasks)]}


@pytest.mark.parametrize("deps", [None, []])
def test_with_environment_is_a_noop_without_dependencies(deps):
    job = with_environment(_job(1), deps)

    assert "environments" not in job
    assert "environment_key" not in job["tasks"][0]


def test_with_environment_declares_the_spec_once_on_the_job():
    deps = ["../dist/*.whl", "https://example.com/mlplatform-1.0.0.whl"]

    job = with_environment(_job(1), deps)

    assert job["environments"] == [
        {"environment_key": ENVIRONMENT_KEY, "spec": {"client": "3", "dependencies": deps}}
    ]


def test_with_environment_applies_the_key_to_every_task():
    """Regressão: monitoring aplicava o environment_key só em tasks[0]. Nunca
    quebrou porque seus jobs têm uma task só — mas num job com duas, a segunda
    subiria sem ambiente e falharia em runtime por dependência ausente."""
    job = with_environment(_job(3), ["../dist/*.whl"])

    assert [t.get("environment_key") for t in job["tasks"]] == [ENVIRONMENT_KEY] * 3


def test_with_environment_copies_the_dependency_list():
    """O chamador não pode ficar aliasado ao recurso gerado."""
    deps = ["../dist/*.whl"]

    job = with_environment(_job(1), deps)
    deps.append("mutacao-depois-da-chamada")

    assert job["environments"][0]["spec"]["dependencies"] == ["../dist/*.whl"]


def test_the_generated_yaml_never_uses_anchors():
    """Os geradores compartilham a lista de job parameters entre os jobs, e o
    dumper padrão transforma a segunda ocorrência em `*id001`. O DABs resolve,
    mas o YAML é a principal superfície de debug desde que os notebooks saíram —
    quem o abre precisa ver o conteúdo, não uma referência."""
    import tempfile
    from pathlib import Path

    from mlplatform.core.resource_gen import dump_yaml

    shared = [{"name": "catalog", "default": "workspace"}]
    resource = {"resources": {"jobs": {"a": {"parameters": shared}, "b": {"parameters": shared}}}}

    out = Path(tempfile.mkdtemp()) / "r.yml"
    dump_yaml(resource, str(out))
    text = out.read_text(encoding="utf-8")

    assert "&id" not in text
    assert "*id" not in text
    assert text.count("name: catalog") == 2
