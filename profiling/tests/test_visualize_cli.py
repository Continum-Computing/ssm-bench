"""
Tests for the visualize.py command line entry point.

These run the script as a subprocess rather than importing it, and that is the
whole point. The module once had no __main__ block and never imported argparse,
so `python profiling/visualize.py <results.json>` exited 0 and wrote nothing.
Importing the module would have looked fine the entire time. Only executing it
the way the documentation says to catches that.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "profiling" / "visualize.py"
A100_RESULTS = (
    REPO_ROOT / "profiling" / "results" / "a100-modal-2026-10-10" / "benchmark_results.json"
)

EXPECTED_PLOTS = (
    "latency_vs_seq_len.png",
    "bandwidth_utilization.png",
    "compute_utilization.png",
    "roofline_model.png",
)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke visualize.py the way a contributor would, from the repo root."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestCommandLineEntryPoint:
    def test_help_renders(self):
        result = run_cli("--help")

        assert result.returncode == 0, result.stderr
        assert "results_path" in result.stdout

    def test_writes_the_layout_the_manifest_describes(self, tmp_path):
        result = run_cli(str(A100_RESULTS), "--run-dir", str(tmp_path))

        assert result.returncode == 0, result.stderr

        for name in EXPECTED_PLOTS:
            assert (tmp_path / "plots" / name).is_file(), f"missing {name}"

        # One summary, at run-dir level. A second copy under plots/ is the
        # defect that let stale GPU peaks survive a correction pass.
        assert (tmp_path / "benchmark_summary.txt").is_file()
        assert not (tmp_path / "plots" / "benchmark_summary.txt").exists()

    def test_run_dir_defaults_to_the_directory_holding_the_json(self, tmp_path):
        copied = tmp_path / "benchmark_results.json"
        shutil.copy(A100_RESULTS, copied)

        result = run_cli(str(copied))

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "benchmark_summary.txt").is_file()
        assert (tmp_path / "plots" / "roofline_model.png").is_file()


class TestArgumentValidation:
    def test_a_missing_results_file_fails_loudly(self, tmp_path):
        result = run_cli(str(tmp_path / "absent.json"))

        assert result.returncode != 0
        assert "no such results file" in result.stderr
