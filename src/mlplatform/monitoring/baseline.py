from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class TrainingRun:
    entity_name: str
    status: str
    window_start: date
    window_end: date
    run_ts: datetime


class NoTrainingRunError(Exception):
    """Levantado quando não há nenhum run SUCCESS de treino registrado para o modelo."""


def resolve_baseline_window(training_runs: list[TrainingRun], full_model_name: str) -> tuple[date, date]:
    candidates = [r for r in training_runs if r.entity_name == full_model_name and r.status == "SUCCESS"]
    if not candidates:
        raise NoTrainingRunError(f"no successful training run found for model '{full_model_name}'")

    most_recent = max(candidates, key=lambda r: r.run_ts)
    return most_recent.window_start, most_recent.window_end
