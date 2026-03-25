import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "y_client" / "clients" / "logging_utils.py"
SPEC = importlib.util.spec_from_file_location("logging_utils", MODULE_PATH)
logging_utils = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(logging_utils)
resolve_log_file_path = logging_utils.resolve_log_file_path


def test_resolve_log_file_path_uses_experiment_directory_for_relative_name(tmp_path):
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    resolved = resolve_log_file_path(str(exp_dir), "reddit_client.log")

    assert Path(resolved) == (exp_dir / "reddit_client.log").resolve()


def test_resolve_log_file_path_keeps_absolute_name(tmp_path):
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    absolute = (tmp_path / "logs" / "reddit_client.log").resolve()

    resolved = resolve_log_file_path(str(exp_dir), str(absolute))

    assert Path(resolved) == absolute
