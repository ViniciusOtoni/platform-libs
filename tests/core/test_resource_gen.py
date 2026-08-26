from pathlib import Path

from mlplatform.core.resource_gen import dump_yaml


def test_dump_yaml_writes_the_resource_dict(tmp_path: Path):
    out = tmp_path / "generated.yml"

    dump_yaml({"resources": {"jobs": {"a_job": {"name": "a_job"}}}}, str(out))

    content = out.read_text(encoding="utf-8")
    assert "resources:" in content
    assert "a_job" in content
