# Results Schema

This document describes the structure of benchmark result files in `profiling/results/`.

## File structure

Each benchmark run produces a directory with this layout:

```
{gpu}-{platform}-{date}/
├── metadata.json              # Run metadata and summary
├── benchmark_results.json     # Raw per-configuration data
├── benchmark_summary.txt      # Human-readable summary
└── plots/
    ├── latency_vs_seq_len.png
    ├── bandwidth_utilization.png
    ├── compute_utilization.png
    └── roofline_model.png
```

## metadata.json

Top-level metadata for a benchmark run.

```json
{
  "run_id": "t4-colab-2026-10-10",
  "timestamp": "2026-10-10T10:45:06.066986",
  "platform": {
    "name": "Google Colab",
    "gpu": {
      "name": "Tesla T4",
      "compute_capability": "7.5",
      "memory_gb": 15.64,
      "peak_bandwidth_gb_s": 900,
      "peak_fp32_tflops": 13.0,
      "peak_bf16_tflops": 26.0
    }
  },
  "software": {
    "python": "3.13.15",
    "pytorch": "2.11.0+cu130",
    "cuda": "13.0"
  },
  "configuration": {
    "sequence_lengths": [512, 1024, 2048, 4096],
    "state_dimensions": [64, 128, 256],
    "input_dim": 256,
    "output_dim": 256,
    "dtype": "bf16",
    "batch_size": 1,
    "warmup_runs": 10,
    "benchmark_runs": 100
  },
  "summary": {
    "total_configurations": 36,
    "models_tested": ["ssm_recurrent", "ssm_scan", "attention"],
    "avg_ssm_recurrent_latency_ms": 319.4,
    "avg_ssm_scan_latency_ms": 413.0,
    "avg_attention_latency_ms": 0.23,
    "avg_ssm_bandwidth_utilization_pct": 0.06,
    "avg_attention_bandwidth_utilization_pct": 0.82,
    "avg_ssm_compute_utilization_pct": 0.01,
    "avg_attention_compute_utilization_pct": 5.1
  },
  "key_findings": "SSM is 100-200x slower than attention...",
  "files": {
    "raw_data": "benchmark_results.json",
    "summary": "benchmark_summary.txt",
    "plots": {
      "latency": "plots/latency_vs_seq_len.png",
      "bandwidth": "plots/bandwidth_utilization.png",
      "compute": "plots/compute_utilization.png",
      "roofline": "plots/roofline_model.png"
    }
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique identifier: `{gpu}-{platform}-{date}` |
| `timestamp` | string | ISO 8601 timestamp of when the benchmark ran |
| `platform.name` | string | Execution platform (Google Colab, Modal, local) |
| `platform.gpu.*` | object | GPU specifications |
| `software.*` | object | Software versions (Python, PyTorch, CUDA) |
| `configuration.*` | object | Benchmark parameters |
| `summary.*` | object | Aggregated statistics across all configurations |
| `key_findings` | string | One-line summary of results |
| `files.*` | object | Pointers to output files |

## benchmark_results.json

Raw per-configuration benchmark data.

```json
{
  "metadata": {
    "timestamp": "2026-10-10T10:45:06.066986",
    "gpu": {
      "name": "Tesla T4",
      "compute_capability": "7.5",
      "total_memory_gb": 15.637,
      "memory_bandwidth_gb_s": 900,
      "fp32_tflops": 13.0,
      "fp16_tflops": 26.0,
      "bf16_tflops": 26.0
    },
    "python_version": "3.13.15",
    "torch_version": "2.11.0+cu130",
    "cuda_version": "13.0"
  },
  "results": [
    {
      "config": {
        "seq_len": 512,
        "batch_size": 1,
        "state_dim": 64,
        "input_dim": 256,
        "output_dim": 256,
        "dtype": "bf16"
      },
      "model_type": "ssm_recurrent",
      "latency_stats": {
        "mean_ms": 109.62,
        "std_ms": 20.67,
        "min_ms": 92.45,
        "max_ms": 183.38,
        "median_ms": 102.72,
        "p95_ms": 160.57,
        "p99_ms": 183.38,
        "runs": 100
      },
      "flops": 132120576,
      "bytes_transferred": 21864576,
      "arithmetic_intensity": 6.04,
      "achieved_bandwidth_gb_s": 0.20,
      "achieved_tflops": 0.0012,
      "bandwidth_utilization_pct": 0.022,
      "compute_utilization_pct": 0.0046
    }
  ]
}
```

### Per-result fields

| Field | Type | Description |
|-------|------|-------------|
| `config.seq_len` | int | Sequence length |
| `config.batch_size` | int | Batch size |
| `config.state_dim` | int | SSM state dimension |
| `config.input_dim` | int | Input dimension |
| `config.output_dim` | int | Output dimension |
| `config.dtype` | string | Data type (bf16, fp16, fp32) |
| `model_type` | string | `ssm_recurrent`, `ssm_scan`, or `attention` |
| `latency_stats.*` | object | Latency statistics over 100 runs |
| `flops` | int | Total FLOPS for this configuration |
| `bytes_transferred` | int | Total bytes moved (weights, activations, state) |
| `arithmetic_intensity` | float | FLOP/byte ratio |
| `achieved_bandwidth_gb_s` | float | Achieved memory bandwidth |
| `achieved_tflops` | float | Achieved compute performance |
| `bandwidth_utilization_pct` | float | Bandwidth utilization as % of peak |
| `compute_utilization_pct` | float | Compute utilization as % of peak |

## Naming convention

Result directories follow `{gpu}-{platform}-{YYYY-MM-DD}`:

- `gpu`: lowercase model name (t4, a100, h100)
- `platform`: execution platform (colab, modal, local)
- `date`: ISO 8601 date

Examples:
- `t4-colab-2026-10-10`
- `a100-modal-2026-10-10`
- `h100-local-2026-11-15`

## Immutability

Once committed, results are never modified. If a bug is found, a new result set is created with a corrected run.
