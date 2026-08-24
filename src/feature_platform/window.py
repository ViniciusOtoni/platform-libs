from datetime import date

from .types import DateRange


class NoCheckpointError(Exception):
    """Levantado quando o modo incremental roda sem nenhum run SUCCESS anterior registrado."""


def resolve_incremental_window(last_success_end: date | None, today: date) -> DateRange:
    if last_success_end is None:
        raise NoCheckpointError(
            "no successful run found for this feature table; run a backfill first "
            "to establish an initial checkpoint before scheduling incremental runs"
        )
    if last_success_end >= today:
        raise ValueError(
            f"nothing to process: checkpoint ({last_success_end}) is not before today ({today})"
        )
    return DateRange(start=last_success_end, end=today)


def parse_backfill_window(start_date: str, end_date: str) -> DateRange:
    return DateRange(start=date.fromisoformat(start_date), end=date.fromisoformat(end_date))
