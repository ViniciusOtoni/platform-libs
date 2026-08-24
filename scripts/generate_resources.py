import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "src"):
    sys.path.insert(0, str(_p))

import examples.serving_configs  # noqa: F401
from serving_platform.resource_gen import write_resources

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_serving.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(str(output_path))
    print(f"resources written to {output_path}")
