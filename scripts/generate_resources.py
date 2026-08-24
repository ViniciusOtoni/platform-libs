import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import examples.features  # noqa: F401  (importa o exemplo para popular o registro)
from feature_platform.resource_gen import write_job_resource

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_feature_pipeline.job.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_job_resource(str(output_path))
    print(f"resource written to {output_path}")
