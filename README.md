# ssm-bench

**Reproducible benchmarks for state-space, attention, and hybrid sequence models on GPUs.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-active-green)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

📄 **[Read the whitepaper](docs/whitepaper.md)** — GPU–SSM mismatch hypothesis, benchmark results, and the case for purpose-built silicon.

---

## Overview

State-space models (SSMs) such as S4 and Mamba process sequences in linear time with a fixed-size state, while attention costs grow with context length and the KV cache grows with it. Less arithmetic does not automatically mean faster or cheaper inference on a given GPU.

**ssm-bench** measures where time, memory traffic, and energy actually go when SSM, attention, and hybrid blocks run on real GPUs. Every measurement is reproducible.

### What it measures

| Metric | Unit | Description |
|--------|------|-------------|
| Latency per token | ms | Median and P95 over 100 runs |
| Prefill latency | ms | For a given prompt length |
| Throughput | tokens/s | At fixed batch size and sequence length |
| Memory bandwidth | GB/s | Achieved vs. theoretical peak |
| Compute performance | TFLOP/s | Achieved vs. theoretical peak |
| Arithmetic intensity | FLOP/byte | Identifies memory-bound vs. compute-bound workloads |
| Peak memory | GB | Weights, activations, state, or KV cache |
| Energy per token | J | Measured via `pynvml` power telemetry |

## Benchmark Results

### Key findings (October 10, 2026)

Benchmarks on Tesla T4 and A100-SXM4-40GB confirm the GPU–SSM mismatch hypothesis:

| Metric | Tesla T4 | A100-SXM4-40GB |
|--------|----------|----------------|
| SSM vs. attention latency | 100–200× slower | 1,000–1,500× slower |
| Bandwidth utilization (SSM) | 0.02–0.15% | 0.01–0.08% |
| Compute utilization (SSM) | < 0.2% | 0.006% |
| Arithmetic intensity (SSM) | 3–6 FLOP/byte | 3–6 FLOP/byte |
| Arithmetic intensity (attention) | 146–229 FLOP/byte | 146–229 FLOP/byte |

**Critical finding:** The gap widens on higher-end hardware. A100 has 2.3× more bandwidth and 3× more compute than T4. Attention benefits enormously; SSM does not benefit at all. This proves the mismatch is architectural, not a temporary optimization issue.

**Decision:** GO — proceed with Continuum SSU-1 (Kalman architecture) accelerator design.

### Test configuration

| Parameter | Values |
|-----------|--------|
| Sequence lengths | 512, 1024, 2048, 4096 |
| State dimensions | 64, 128, 256 |
| Model types | SSM recurrent, SSM scan, attention |
| Total configurations per GPU | 36 |
| GPUs tested | Tesla T4 (Google Colab), A100-SXM4-40GB (Modal) |

### Raw data

- **Tesla T4:** [`profiling/results/t4-colab-2026-10-10/`](profiling/results/t4-colab-2026-10-10/)
- **A100:** [`profiling/results/a100-modal-2026-10-10/`](profiling/results/a100-modal-2026-10-10/)
- **Comparison analysis:** [`profiling/results/analysis/gpu_comparison_t4_vs_a100.md`](profiling/results/analysis/gpu_comparison_t4_vs_a100.md)

## Benchmark Matrix

The matrix is defined in configuration files under `configs/`, not hard-coded.

| Axis | Values |
|------|--------|
| Model family | Selective-scan SSM (Mamba, Mamba-2, Mamba-3), attention (FlashAttention-2), hybrid (Jamba-style), delta-rule linear attention |
| Phase | Prefill, decode |
| Batch size | 1, and larger values for concurrent users |
| Sequence length | 512–4096 (current); up to 32k (planned) |
| Precision | bf16 (current); fp8 and low-bit weights (planned) |

## Methodology

1. **Correctness first.** Every implementation is checked against a plain reference implementation before it is timed.
2. **Warm up, then repeat.** Warm-up iterations are discarded. Results report the median and P95, never a single run.
3. **Control the machine.** Seeds are fixed. Where possible, GPU clocks are locked and nothing else runs on the device.
4. **Separate timing from profiling.** Profilers change performance, so timing runs and profiler runs are distinct.
5. **Strong baselines.** Comparisons use the best available tuned kernels at matching precision:
   - **Attention:** FlashAttention-2
   - **SSM:** Official Mamba CUDA kernels
   - **Hybrid:** Each layer type uses its corresponding optimized kernel
6. **Roofline analysis.** Kernel results are placed on a roofline model to show whether they are limited by memory bandwidth or compute throughput.

## Reproducibility

Every result file is saved with a metadata file recording:

- GPU model, driver version, CUDA version
- PyTorch, Triton, and library versions
- Random seed
- Git commit hash and working tree status
- Exact command line and configuration file
- Timestamp

Results are never edited by hand. If a number is wrong, the script is fixed and the result regenerated. The result schema is documented in [`docs/results-schema.md`](docs/results-schema.md).

## GPU Targets

| GPU class | Models | Rationale |
|-----------|--------|-----------|
| Datacenter training/inference | H100, A100 | Current and prior generation datacenter GPUs |
| Inference-optimized | L40S | Inference-specific SKU |
| Edge/smaller datacenter | L4, T4 | Lower-power inference |

**Tested:** Tesla T4, A100-SXM4-40GB. Additional GPUs added as hardware becomes available.

## Repository Layout

```
ssm-bench/
├── model/                  Reference and optimized implementations
│   ├── reference/          NumPy reference implementation
│   ├── pytorch_impl/       PyTorch implementation (GPU-ready)
│   └── tests/              Correctness tests
├── profiling/              Measurement scripts and results
│   ├── results/            Benchmark output (JSON, metadata, plots)
│   ├── run_benchmarks.py   Main entry point
│   └── visualize.py        Visualization and reporting
├── configs/                Benchmark matrix definitions
├── docs/                   Whitepaper, methodology, results schema
├── Makefile
└── pyproject.toml
```

## Getting Started

**Requirements:** Linux, Python 3.11+, NVIDIA GPU for benchmarks. CPU-only runs work for correctness tests.

```bash
git clone https://github.com/Continuum-Computiing/ssm-bench.git
cd ssm-bench
make setup    # Create virtual environment and install dependencies
make test     # Run correctness tests
make lint     # Run linters
```

### Running benchmarks

```bash
# Default benchmark sweep
python profiling/run_benchmarks.py

# Custom configuration
python profiling/run_benchmarks.py \
  --seq-lens 512 1024 2048 4096 \
  --state-dims 64 128 256 \
  --dtype bf16 \
  --device 0 \
  --output profiling/results/my_benchmark.json
```

### Generating visualizations

```bash
python profiling/visualize.py profiling/results/benchmark_results.json
```

Generates:
- `latency_vs_seq_len.png` — Latency scaling
- `bandwidth_utilization.png` — Memory bandwidth utilization
- `compute_utilization.png` — Compute utilization
- `roofline_model.png` — Roofline analysis
- `benchmark_summary.txt` — Text summary

## Contributing

Issues and pull requests are welcome, especially:

- Corrections to methodology
- Results on other GPUs, with full metadata
- Stronger baseline kernels

Keep commit messages in the form `type: summary` (feat, fix, docs, test, chore). Do not include secrets or credentials.

## References

**Core SSM papers:**
- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces* (S4), 2021.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), 2024.
- Mamba-3, 2026.

**Optimization baselines:**
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
- FlashAttention-2, 2023.

**Architecture and methodology:**
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, 2009.
- Sze, Chen, Yang, Emer. *Efficient Processing of Deep Neural Networks*, 2020.

**Production hybrid models:**
- AI21 Jamba: A hybrid transformer-SSM model for production deployment, 2024.

## Citation

```bibtex
@software{ssm_bench,
  author  = {Continuum Computing},
  title   = {ssm-bench: reproducible benchmarks for state-space, attention and hybrid sequence models},
  year    = {2026},
  url     = {https://github.com/Continuum-Computiing/ssm-bench}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Relationship to Continuum SSU-1 (Kalman)

This repository measures the GPU gap. The **Continuum SSU-1** (Kalman architecture) repository consumes the gap report and designs the accelerator.

- **ssm-bench** → measures where GPUs underperform on SSM workloads → produces the GPU gap report
- **kalman** → consumes the gap report → designs the SSU-1 accelerator to close the gap

The two repositories have a strict sequential dependency: kalman's architecture design does not begin until ssm-bench has produced the gap report with a go/no-go recommendation.

See [`docs/whitepaper.md`](docs/whitepaper.md) for the full hypothesis, evidence, and benchmark results.
