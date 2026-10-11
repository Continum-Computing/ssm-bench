# ssm-bench

**Reproducible benchmarks for state-space, attention and hybrid sequence models on GPUs.**

[![CI](https://github.com/[ORG]/ssm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/[ORG]/ssm-bench/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-early-orange)

> **Status: early.** The scaffolding and methodology are in place. **No benchmark results have been published yet.** Numbers appear in this README only after they can be reproduced from the scripts in this repository.

---

## What this is

State-space models (SSMs) such as S4 and Mamba process sequences in linear time with a fixed-size state, while attention costs grow with context length and the key-value cache grows with it. Less arithmetic does not automatically mean faster or cheaper inference on a given GPU.

`ssm-bench` measures where time, memory traffic and energy actually go when SSM, attention and hybrid blocks run on real GPUs, and keeps every measurement reproducible.

It is built to answer questions like:

- At batch size 1 decode, how much of a layer's time is spent moving state and weights compared with computing?
- How do the memory footprint of an SSM's state and a transformer's KV cache compare as context length and concurrent users grow?
- How far below the roofline do these kernels run, and why?
- Where does a hybrid model land between the two?

## What it measures

| Metric | Unit | Notes |
| --- | --- | --- |
| Latency per token (decode) | ms | Median and spread over repeated runs |
| Prefill latency | ms | For a given prompt length |
| Throughput | tokens/s | At fixed batch size and sequence length |
| Achieved memory bandwidth | GB/s | From profiler counters, reported against the device peak |
| Achieved compute | TFLOP/s | Reported against the device peak at the same precision |
| Arithmetic intensity | FLOP/byte | Used to place each kernel on a roofline |
| Peak memory | GB | Weights, activations, state or KV cache |
| State size and KV cache size | MB per sequence | The memory a model needs per user |
| Kernel count and launch overhead | count, µs | Matters at small batch sizes |
| Energy per token | J | Measured via `pynvml` (NVIDIA Management Library) power telemetry. When telemetry is unavailable, the field is set to `null` and a `power_measurement_available: false` flag is included in the metadata file. External power measurement (e.g., watt meter) is not used in v1. |

## Benchmark matrix

The matrix is defined in configuration files under `configs/`, not hard-coded.

| Axis | Values (planned) |
| --- | --- |
| Model family | Selective-scan SSM block (Mamba-style), attention block, hybrid block (Jamba-style with interleaved SSM and attention layers), delta-rule linear attention, test-time training architectures |
| Phase | Prefill, decode |
| Batch size | 1, and larger values for concurrent users |
| Sequence length | A sweep from short to long context |
| Precision | bf16 first; fp8 and low-bit weights where supported |

**Model family notes:**
- **Selective-scan SSM:** Mamba, Mamba-2, Mamba-2S (2026 simplified variant), Mamba-3 (2026)
- **Attention:** Standard multi-head attention with FlashAttention-style optimization
- **Hybrid:** Jamba-style architectures where SSM and attention layers are interleaved — these are production-deployed models (AI21) and represent the realistic workload a future accelerator must handle
- **Delta-rule linear attention:** The convergence point the field reached in 2025-2026, where linear attention with test-time training merges with SSM-style recurrence
- **Test-time training:** Architectures where the recurrent state is an online-learned compressed memory

## Methodology

1. **Correctness first.** Every implementation is checked against a plain reference implementation before it is timed.
2. **Warm up, then repeat.** Warm-up iterations are discarded. Results report the median and the spread, never a single run.
3. **Control the machine.** Seeds are fixed. Where possible, GPU clocks are locked and nothing else runs on the device.
4. **Separate timing from profiling.** Profilers change performance, so timing runs and profiler runs are distinct.
5. **Strong baselines.** Comparisons use the best available tuned kernels at matching precision:
   - **Attention:** FlashAttention-2 (the production-grade optimized kernel)
   - **SSM:** Official Mamba CUDA kernels from the Mamba repository (the selective scan implementation from the authors)
   - **Hybrid:** Each layer type uses its corresponding optimized kernel
   - Comparisons are never against deliberately weak or unoptimized code. If a tuned kernel does not exist for a variant, this is flagged in the result metadata.
6. **Roofline analysis.** Kernel results are placed on a roofline model to show whether they are limited by memory bandwidth or by compute throughput.

## Reproducibility

Every result file is saved with a metadata file next to it recording:

- GPU model, driver version and CUDA version
- Versions of PyTorch, Triton and other libraries
- Random seed
- Git commit hash and whether the working tree was clean
- The exact command line and configuration file
- Date and time

Results are never edited by hand. If a number is wrong, the script is fixed and the result regenerated. The result schema is documented in `docs/results-schema.md`.

## GPU targets

Profiling targets NVIDIA GPUs representative of datacenter inference workloads:

| GPU class | Specific models (planned) | Rationale |
| --- | --- | --- |
| Datacenter training/inference | H100, A100 | Current and prior generation datacenter GPUs; the primary deployment target for large models |
| Inference-optimized | L40S | Inference-specific SKU; represents the "inference at scale" deployment pattern |
| Edge/smaller datacenter | L4, T4 | Lower-power inference; relevant if SSMs target edge deployment |

The first round of profiling will target H100 and A100. Additional GPUs are added as hardware becomes available. CPU-only runs are supported for correctness tests but not for performance profiling.

## GPU gap report

The primary deliverable of this repository is the **GPU gap report** — a structured analysis of where GPUs underperform on SSM workloads relative to their theoretical capability.

The gap report lives in `docs/gap-report/` and contains:

1. **Per-kernel analysis:** For each kernel type (SSM selective scan, attention, hybrid layer), where time and memory bandwidth are spent, and how far below the roofline the kernel operates.
2. **Workload-level analysis:** End-to-end layer latency, throughput, and energy for SSM vs. attention vs. hybrid at matched model quality.
3. **Scaling analysis:** How the gap changes with batch size, sequence length, and model size.
4. **Go/no-go recommendation:** Based on the measured gap, whether the evidence supports custom silicon (large gap) or suggests the project should pivot (small gap).

The gap report is consumed by the `kalman` repository, which designs the accelerator architecture based on the measured inefficiencies. The result format is structured (JSON schema in `docs/results-schema.md`) so that `kalman`'s architecture cost model can programmatically score candidate designs against measured GPU performance.

## Repository layout

Folders appear as the work reaches them.

```
ssm-bench/
├── model/            Reference and optimized implementations (NumPy, PyTorch, Triton)
│   └── tests/        Correctness tests against the reference
├── profiling/        Measurement scripts
│   └── results/      Result files and their metadata files
├── configs/          Benchmark matrix definitions
├── docs/             Methodology notes, results schema, lab notebook
├── Makefile
└── pyproject.toml
```

## Getting started

**Requirements:** Linux, Python 3.11 or newer, and an NVIDIA GPU for GPU benchmarks. CPU-only runs work for the correctness tests.

```bash
git clone https://github.com/[ORG]/ssm-bench.git
cd ssm-bench
make setup    # create the virtual environment and install dependencies
make test     # run the correctness tests
make lint     # run linters
```

The command for running a benchmark is documented here when the first benchmark lands.

## Results

No results have been published yet. When they are, they will appear in this format:

| Model | Phase | Batch | Seq. length | Precision | Latency (ms) | Bandwidth (GB/s) | GPU | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Hardware tested**

| GPU | Memory | Driver | CUDA | Tested by |
| --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD |

## Limitations

Readers should treat results with these in mind:

- Layer and block microbenchmarks do not predict end-to-end serving performance.
- Results depend on the specific kernels, library versions and GPU used. They do not generalize to every device.
- Power measurement through GPU telemetry is approximate.
- Only NVIDIA GPUs are covered at first.
- Fused, hand-tuned kernels change the picture. The kernels used are always named alongside results.

## Adding a model or kernel

1. Add the implementation under `model/`.
2. Add a test that compares it against the reference implementation.
3. Add or extend a configuration under `configs/`.
4. Run it, and commit the result file and its metadata file together.
5. Open a pull request describing what was added and what it was compared against.

## Contributing

Issues and pull requests are welcome, especially:

- Corrections to methodology
- Results on other GPUs, with full metadata
- Stronger baseline kernels

Please keep commit messages in the form `type: summary` (feat, fix, docs, test, chore). Do not include secrets, credentials or private data in contributions.

## References

**Core SSM papers (read in this order):**
- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces* (S4), 2021.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), 2024.
- Mamba-3, 2026. [Citation pending — paper published, update with full reference when available]
- Mamba-2S (simplified variant), 2026. [Citation pending]

**Convergence and related work:**
- Linear attention with delta-rule recurrence and test-time training, 2025-2026. [Survey paper citation pending — this is the convergence point where linear attention and SSM-style recurrence merge into a single research direction]

**Known SSM limitations:**
- NVIDIA research on Mamba's weaknesses: lag on in-context learning and copying/lookup tasks compared to transformers. [Citation pending — important for understanding architectural tradeoffs]

**Optimization baselines:**
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
- FlashAttention-2, 2023.

**Architecture and methodology:**
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, 2009.
- Sze, Chen, Yang, Emer. *Efficient Processing of Deep Neural Networks*, 2020. (For understanding hardware-algorithm co-design)

**Production hybrid models:**
- AI21 Jamba: A hybrid transformer-SSM model for production deployment, 2024. [Citation pending — represents the realistic workload a future accelerator must handle]

## Citation

If you use this repository, please cite it. A `CITATION.cff` file will be added with the first release.

```bibtex
@software{ssm_bench,
  author  = {[OWNER NAME]},
  title   = {ssm-bench: reproducible benchmarks for state-space, attention and hybrid sequence models},
  year    = {[YEAR]},
  url     = {https://github.com/[ORG]/ssm-bench}
}
```

## License

[LICENSE NAME]. See `LICENSE`. Decide the license before making this repository public.

## Maintainer

[OWNER NAME], [CONTACT EMAIL]

## Relationship to kalman

This repository (`ssm-bench`) is the benchmark suite that measures the GPU gap. The `kalman` repository consumes the gap report and designs the accelerator architecture.

- **ssm-bench** → measures where GPUs underperform on SSM workloads → produces the GPU gap report
- **kalman** → consumes the gap report → designs the SSU accelerator to close the gap

The two repositories have a strict sequential dependency: kalman's architecture design does not begin until ssm-bench has produced the gap report with a go/no-go recommendation. The result format is structured (JSON schema in `docs/results-schema.md`) so that kalman can programmatically consume the data.

See `kalman/docs/whitepaper.md` for the full hypothesis, evidence, and benchmark results.