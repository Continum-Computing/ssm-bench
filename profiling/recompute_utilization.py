"""
Recompute utilization percentages for already-collected benchmark runs.

Why this exists
---------------
The 2026-10-10 T4 and A100 runs divided measured throughput by the wrong
hardware peaks. `get_gpu_specs` looked peaks up by compute capability, and the
Tesla T4 shares capability 7.5 with the RTX 2080 Ti, so the T4 run was scored
against 900 GB/s and 13.0/26.0 TFLOPS instead of its real 320 GB/s and 8.1
TFLOPS. The A100 run was scored against 2039 GB/s (the 80GB card) and a 78.0
TFLOPS non-tensor BF16 rate.

The measurements themselves were always fine. `achieved_bandwidth_gb_s`,
`achieved_tflops`, `latency_stats`, `flops` and `bytes_transferred` are
untouched by this script. Only the two derived percentages and the peak
constants they were divided by change.

What it does
------------
For each result directory:
  1. Looks up correct datasheet peaks by GPU name (profiling/gpu_specs_db.py).
  2. Recomputes bandwidth_utilization_pct and compute_utilization_pct.
  3. Infers how each `summary` average was grouped, by reproducing the stored
     value from the original peaks. It applies the same grouping to the
     corrected peaks rather than guessing.
  4. Writes a `peak_correction` audit block recording the old peaks, the new
     peaks, and the date, so the change is traceable.
  5. Regenerates plots and the text summary.

Measured values are never rewritten. If a field this script expects is missing
it raises rather than filling in a plausible number.

Usage
-----
    python profiling/recompute_utilization.py                 # all result sets
    python profiling/recompute_utilization.py --dry-run       # report only
    python profiling/recompute_utilization.py --run-id t4-colab-2026-10-10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profiling.gpu_specs_db import lookup_peak_specs, peak_for_dtype, specs_to_dict

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CORRECTION_DATE = "2026-10-13"


def _mean(values: List[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty group")
    return sum(values) / len(values)


def _grouping_for(results: List[Dict], field: str, stored: float) -> Optional[str]:
    """
    Work out which models a stored summary average was computed over.

    Recomputes the average from the original per-config values under each
    candidate grouping and returns the one that reproduces the stored number.
    Returns None if nothing matches, so the caller can report it instead of
    silently picking one.
    """
    candidates = {
        "ssm_recurrent": ["ssm_recurrent"],
        "ssm_recurrent+ssm_scan": ["ssm_recurrent", "ssm_scan"],
        "attention": ["attention"],
    }
    for label, model_types in candidates.items():
        values = [r[field] for r in results if r["model_type"] in model_types]
        if not values:
            continue
        if abs(_mean(values) - stored) < 5e-3:
            return label
    return None


def correct_result_set(run_dir: Path, dry_run: bool = False) -> Dict:
    """Recompute derived metrics for one result directory."""
    raw_path = run_dir / "benchmark_results.json"
    meta_path = run_dir / "metadata.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"no benchmark_results.json in {run_dir}")

    raw = json.loads(raw_path.read_text())
    results = raw["results"]

    gpu_meta = raw["metadata"]["gpu"]
    gpu_name = gpu_meta["name"]

    specs = lookup_peak_specs(gpu_name)
    if specs is None:
        raise ValueError(
            f"{run_dir.name}: no datasheet peaks for '{gpu_name}'. "
            "Add it to profiling/gpu_specs_db.py first."
        )

    old_peaks = {
        "memory_bandwidth_gb_s": gpu_meta["memory_bandwidth_gb_s"],
        "fp32_tflops": gpu_meta["fp32_tflops"],
        "fp16_tflops": gpu_meta.get("fp16_tflops"),
        "bf16_tflops": gpu_meta["bf16_tflops"],
    }

    changed = 0
    # Snapshot the stored percentages before mutating anything. The summary
    # averages in metadata.json were computed from these values under the old
    # peaks, so reproducing them is the only way to learn which models each
    # average was grouped over.
    original = [
        {
            "model_type": r["model_type"],
            "bandwidth_utilization_pct": r["bandwidth_utilization_pct"],
            "compute_utilization_pct": r["compute_utilization_pct"],
        }
        for r in results
    ]

    for r in results:
        dtype = r["config"]["dtype"]
        peak_compute = peak_for_dtype(specs, dtype)

        new_bw = (r["achieved_bandwidth_gb_s"] / specs.memory_bandwidth_gb_s) * 100
        new_compute = (r["achieved_tflops"] / peak_compute) * 100

        if (abs(new_bw - r["bandwidth_utilization_pct"]) > 1e-9
                or abs(new_compute - r["compute_utilization_pct"]) > 1e-9):
            changed += 1

        if not dry_run:
            r["bandwidth_utilization_pct"] = new_bw
            r["compute_utilization_pct"] = new_compute

    report = {
        "run_id": run_dir.name,
        "gpu": gpu_name,
        "configurations": len(results),
        "rows_changed": changed,
        "old_peaks": old_peaks,
        "new_peaks": specs_to_dict(specs),
    }

    if dry_run:
        return report

    raw["metadata"]["gpu"].update({
        "memory_bandwidth_gb_s": specs.memory_bandwidth_gb_s,
        "fp32_tflops": specs.fp32_tflops,
        "fp16_tflops": specs.fp16_tensor_tflops,
        "bf16_tflops": specs.bf16_tflops,
        "specs_source": "datasheet",
        "specs_note": specs.note,
    })

    # Preserve an existing audit block. Reading old_peaks from metadata that a
    # previous run already corrected would record the corrected values as
    # "previous", silently destroying the provenance this block exists to keep.
    prior = raw["metadata"].get("peak_correction")
    raw["metadata"]["peak_correction"] = prior or {
        "date": CORRECTION_DATE,
        "reason": (
            "Peaks were originally looked up by compute capability. The Tesla T4 "
            "shares capability 7.5 with the RTX 2080 Ti and inherited its specs; "
            "the A100 40GB run used the 80GB bandwidth and a non-tensor BF16 rate. "
            "Measured values are unchanged; only derived utilization percentages "
            "and the peaks they are divided by were corrected."
        ),
        "previous_peaks": old_peaks,
        "corrected_peaks": specs_to_dict(specs),
    }
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")

    if meta_path.exists():
        _correct_metadata_file(meta_path, raw, specs, old_peaks, results, original)

    _regenerate_outputs(run_dir, raw)

    return report


def _correct_metadata_file(meta_path: Path, raw: Dict, specs, old_peaks: Dict,
                           results: List[Dict], original: List[Dict]) -> None:
    """
    Rebuild the metadata.json summary block from the raw per-config data.

    Every summary field is recomputed rather than patched. That matters here:
    the T4 run's stored summary did not match its own raw data at all. It held
    the A100's averages (319.4 / 413.0 / 0.23 ms), which is where the claim
    "SSM latency is identical on both GPUs" came from. The T4's real averages
    are 388.3 / 497.9 / 3.247 ms.

    Grouping convention, matching the original schema:
      avg_ssm_*       -> ssm_recurrent only
      avg_attention_* -> attention
      avg_ssm_recurrent_latency_ms / avg_ssm_scan_latency_ms are per model.
    """
    meta = json.loads(meta_path.read_text())
    gpu_block = meta["platform"]["gpu"]

    gpu_block.update({
        "peak_bandwidth_gb_s": specs.memory_bandwidth_gb_s,
        "peak_fp32_tflops": specs.fp32_tflops,
        "peak_fp16_tflops": specs.fp16_tensor_tflops,
        "peak_bf16_tflops": specs.bf16_tflops,
        "specs_source": "datasheet",
        "specs_note": specs.note,
    })

    def subset(*model_types):
        return [r for r in results if r["model_type"] in model_types]

    def avg(rs, field):
        if not rs:
            return None
        return sum(r[field] for r in rs) / len(rs)

    def avg_latency(rs):
        if not rs:
            return None
        return sum(r["latency_stats"]["mean_ms"] for r in rs) / len(rs)

    recurrent = subset("ssm_recurrent")
    scan = subset("ssm_scan")
    attention = subset("attention")

    old_summary = dict(meta.get("summary", {}))

    recomputed = {
        "total_configurations": len(results),
        "models_tested": sorted({r["model_type"] for r in results}),
        "avg_ssm_recurrent_latency_ms": round(avg_latency(recurrent), 3),
        "avg_ssm_scan_latency_ms": round(avg_latency(scan), 3),
        "avg_attention_latency_ms": round(avg_latency(attention), 3),
        "avg_ssm_bandwidth_utilization_pct": round(avg(recurrent, "bandwidth_utilization_pct"), 4),
        "avg_attention_bandwidth_utilization_pct": round(avg(attention, "bandwidth_utilization_pct"), 4),
        "avg_ssm_compute_utilization_pct": round(avg(recurrent, "compute_utilization_pct"), 5),
        "avg_attention_compute_utilization_pct": round(avg(attention, "compute_utilization_pct"), 4),
    }

    # Report which stored values disagreed with the raw data, so the repair is
    # auditable rather than silent.
    contradicted = {}
    for key, new in recomputed.items():
        old = old_summary.get(key)
        if old is None or new is None:
            continue
        if isinstance(new, float) and abs(old - new) <= max(0.01, abs(new) * 0.01):
            continue
        contradicted[key] = {"stored": old, "recomputed": new}

    summary = dict(old_summary)
    summary.update({k: v for k, v in recomputed.items() if v is not None})
    meta["summary"] = summary

    meta["summary_repair"] = {
        "date": CORRECTION_DATE,
        "grouping_convention": {
            "avg_ssm_*": "ssm_recurrent only",
            "avg_attention_*": "attention only",
        },
        "fields_that_contradicted_raw_data": contradicted,
        "note": (
            "Recomputed from benchmark_results.json. Fields listed above held "
            "values that did not match this run's own raw data."
        ),
    }

    meta["peak_correction"] = {
        "date": CORRECTION_DATE,
        "reason": raw["metadata"]["peak_correction"]["reason"],
        "previous_peaks": old_peaks,
        "corrected_peaks": specs_to_dict(specs),
    }

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return contradicted


def _regenerate_outputs(run_dir: Path, raw: Dict) -> None:
    """Rebuild plots and the text summary from the corrected results."""
    from profiling.visualize import (
        generate_summary_report,
        plot_bandwidth_utilization,
        plot_compute_utilization,
        plot_latency_vs_sequence_length,
        plot_roofline_model,
    )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_latency_vs_sequence_length(raw, str(plots_dir / "latency_vs_seq_len.png"))
    plot_bandwidth_utilization(raw, str(plots_dir / "bandwidth_utilization.png"))
    plot_compute_utilization(raw, str(plots_dir / "compute_utilization.png"))
    plot_roofline_model(raw, str(plots_dir / "roofline_model.png"))
    generate_summary_report(raw, str(run_dir / "benchmark_summary.txt"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run-id", help="correct a single result set")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    if args.run_id:
        run_dirs = [RESULTS_DIR / args.run_id]
    else:
        run_dirs = sorted(
            p for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "benchmark_results.json").exists()
        )

    if not run_dirs:
        print("no result sets found", file=sys.stderr)
        return 1

    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"missing result set: {run_dir}", file=sys.stderr)
            return 1

        report = correct_result_set(run_dir, dry_run=args.dry_run)

        prefix = "would correct" if args.dry_run else "corrected"
        print(f"{prefix} {report['run_id']} ({report['gpu']})")
        print(f"  rows: {report['configurations']}, changed: {report['rows_changed']}")
        print(f"  bandwidth peak: {report['old_peaks']['memory_bandwidth_gb_s']} "
              f"-> {report['new_peaks']['memory_bandwidth_gb_s']} GB/s")
        print(f"  bf16 peak:      {report['old_peaks']['bf16_tflops']} "
              f"-> {report['new_peaks']['bf16_tflops']} TFLOPS")
        print(f"  note: {report['new_peaks']['note']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
