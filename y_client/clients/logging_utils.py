from pathlib import Path


def resolve_log_file_path(data_base_path, log_file):
    """Resolve client log files relative to the experiment directory."""
    log_path = Path(log_file)
    if log_path.is_absolute():
        return str(log_path)

    base_path = Path(data_base_path or ".")
    if base_path.suffix:
        base_path = base_path.parent
    return str((base_path / log_path).resolve())
