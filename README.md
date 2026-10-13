# ssm-bench

Benchmarks for state-space models, attention, and hybrid sequence models on NVIDIA GPUs. Measures latency, memory bandwidth, compute utilization, and arithmetic intensity. Everything is reproducible.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-active-green)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

📄 [Read the whitepaper](docs/whitepaper.md)

---

## What this does

State-space models (S4, Mamba, Mamba-2, Mamba-3) process sequences in O(N) with a fixed-size recurrent state. Attention is O(N²) and the KV cache grows with context length. On paper, SSMs should be faster and cheaper.

In practice, GPUs are built for dense matrix multiplication. Attention is a bunch of large matmuls. SSMs are a stream of small state updates. The GPU runs both correctly, but we suspect it is the wrong shape of hardware for SSMs.

That suspicion is the hypothesis this repo exists to test: do GPUs turn the SSM's arithmetic savings into proportional speed and energy gains? We benchmark SSM, attention, and hybrid blocks on real GPUs, keep every result reproducible, and publish the raw data.

Where we are now: the published results come from our own reference implementations, not from tuned kernels. They are a starting point, not the answer.

## Results so far

We ran benchmarks on two GPUs: Tesla T4 (via Google Colab) and A100-SXM4-40GB (via Modal). 36 configurations per GPU (4 sequence lengths × 3 state dimensions × 3 model types).

**What was measured.** Our own reference implementations. The SSM path is a Python for-loop over time steps, which means one small kernel launch per step. The attention path is a plain matmul-softmax-matmul. Batch size 1, bf16. No fused kernels, no CUDA graphs.

| | Tesla T4 | A100 |
|---|---|---|
| SSM vs attention latency | 100-200x slower | 1,000-1,500x slower |
| SSM bandwidth utilization | 0.02-0.15% | 0.01-0.08% |
| SSM compute utilization | < 0.2% | 0.006% |
| SSM arithmetic intensity | 3-6 FLOP/byte | 3-6 FLOP/byte |
| Attention arithmetic intensity | 146-229 FLOP/byte | 146-229 FLOP/byte |

**What these numbers say.** At batch size 1 with a Python loop in the critical path, kernel launch overhead dominates the SSM measurement. Both sides sit far below the hardware limits, which is why the utilization columns are near zero for both. The arithmetic intensity rows carry more weight, since intensity is a property of the algorithm rather than of our implementation.

**What these numbers do not say.** They do not measure an architectural mismatch. This is naive SSM against naive attention, so most of the gap is our code, not the GPU. A fused Mamba kernel and FlashAttention-2 would both land somewhere completely different, and neither is in this table yet. The same caveat covers the "gap widens on better hardware" observation: a workload bound by launch overhead is not expected to benefit from extra bandwidth on any GPU.

**Go/no-go decision: pending.** We are not going to justify custom silicon on a comparison of naive implementations. The decision waits on strong baselines: FlashAttention-2 for attention, official Mamba CUDA kernels for the SSM path, and CUDA graph capture to take launch overhead out of the decode measurement. Architecture work in the private `kalman` repo continues in parallel, but nothing downstream should treat this table as the evidence that settled it.

### Raw data

- [Tesla T4 results](profiling/results/t4-colab-2026-10-10/)
- [A100 results](profiling/results/a100-modal-2026-10-10/)
- [T4 vs A100 comparison](profiling/results/analysis/gpu_comparison_t4_vs_a100.md)

### Visualizations

**Latency scaling** - naive SSM vs naive attention across sequence lengths:

![Latency vs sequence length](profiling/results/t4-colab-2026-10-10/plots/latency_vs_seq_len.png)

**Bandwidth utilization** - neither implementation comes close to the T4's 320 GB/s peak:

![Bandwidth utilization](profiling/results/t4-colab-2026-10-10/plots/bandwidth_utilization.png)

**Compute utilization** - at batch size 1 the GPU spends most of its time waiting between kernel launches:

![Compute utilization](profiling/results/t4-colab-2026-10-10/plots/compute_utilization.png)

**Roofline analysis** - where each kernel sits against the bandwidth and compute limits:

![Roofline model](profiling/results/t4-colab-2026-10-10/plots/roofline_model.png)

## What we measure

| Metric | Unit | Notes |
|---|---|---|
| Latency per token | ms | Median and P95 over 100 runs |
| Prefill latency | ms | For a given prompt length |
| Throughput | tokens/s | Fixed batch size and sequence length |
| Memory bandwidth | GB/s | Achieved vs theoretical peak |
| Compute | TFLOP/s | Achieved vs theoretical peak |
| Arithmetic intensity | FLOP/byte | Tells you if you're memory-bound or compute-bound |
| Peak memory | GB | Weights, activations, state, KV cache |
| Energy per token | J | Planned, via pynvml power telemetry |

The results published so far cover latency, bandwidth, compute utilization, and arithmetic intensity. Energy per token is specified but not implemented, which is why no energy number appears anywhere in this repo. Until it does, any throughput-per-watt statement is a target, not a measurement.

## Benchmark matrix

Passed as command-line arguments to `profiling/run_benchmarks.py`, with the defaults shown below. `configs/` is reserved for config files but is empty today, so nothing is loaded from it yet.

| Axis | Values |
|---|---|
| Model family | Implemented: selective-scan SSM (recurrent and scan) and multi-head attention, both as reference implementations. Planned: Mamba-2, Mamba-3, FlashAttention-2, hybrid (Jamba-style), delta-rule linear attention |
| Phase | Implemented: one full forward pass. Planned: decode-step timing |
| Batch size | 1 only. Larger batches are planned, since serving workloads do not run at batch 1 |
| Sequence length | 512-4096 (done), up to 32k (planned) |
| Precision | bf16 (used in the published results). The runner accepts fp32 and fp16. fp8 is planned |

## How we do it

1. **Correctness first.** Every implementation gets checked against a plain NumPy reference before we time anything.
2. **Warm up, then measure.** We throw away warmup iterations. Results are median and P95, never a single lucky run.
3. **Control the machine.** Fixed seeds. GPU clocks locked when possible. Nothing else on the device.
4. **Timing and profiling are separate.** Profilers change performance, so we keep them apart.
5. **Strong baselines.** The comparison has to be against the best tuned kernels available, not deliberately weak code. This is the part that is not finished:
   - Attention: FlashAttention-2 (not integrated yet)
   - SSM: official Mamba CUDA kernels (not integrated yet)
   - Hybrid: optimized kernel per layer type (not started)
   - Decode: CUDA graph capture, to take launch overhead out of the measurement (not started)

   Until those land, the published numbers describe our reference implementations and nothing else.
6. **Roofline analysis.** Every kernel gets placed on a roofline plot so you can see whether it's bandwidth-limited or compute-limited.

## Reproducibility

Every result file ships with a metadata file next to it:

- GPU model, driver version, CUDA version
- PyTorch version, Triton version, everything else
- Random seed
- Git commit hash, clean working tree
- Exact command line and config file
- Timestamp

Results don't get edited by hand. If something's wrong, we fix the script and regenerate. Schema is in [docs/results-schema.md](docs/results-schema.md).

## GPU targets

| Class | Models | Why |
|---|---|---|
| Datacenter | H100, A100 | Current and prior gen datacenter |
| Inference-optimized | L40S | Inference at scale |
| Edge | L4, T4 | Lower power |

Tested so far: T4, A100. More GPUs as we get access.

## Repo layout

```
ssm-bench/
├── model/                  Implementations
│   ├── reference/          NumPy reference
│   ├── pytorch_impl/       PyTorch (GPU-ready)
│   └── tests/              Correctness tests
├── profiling/              Benchmarks
│   ├── results/            Output (JSON, metadata, plots)
│   ├── run_benchmarks.py   Entry point
│   └── visualize.py        Plots and reports
├── configs/                Benchmark matrix
├── docs/                   Whitepaper, methodology, schema
├── Makefile
└── pyproject.toml
```

## Getting started

Linux, Python 3.11+, NVIDIA GPU for benchmarks. CPU-only works for correctness tests.

```bash
git clone https://github.com/Continum-Computing/ssm-bench.git
cd ssm-bench
make setup
make test
make lint
```

Run benchmarks:

```bash
# Default sweep
python profiling/run_benchmarks.py

# Custom
python profiling/run_benchmarks.py \
  --seq-lens 512 1024 2048 4096 \
  --state-dims 64 128 256 \
  --dtype bf16 \
  --device 0 \
  --output profiling/results/my_benchmark.json
```

Generate plots:

```bash
python profiling/visualize.py profiling/results/benchmark_results.json
```

## Contributing

Issues and PRs welcome. Corrections to methodology, results on other GPUs, stronger baseline kernels.

Commit messages: `type: summary` (feat, fix, docs, test, chore). No secrets.

## References

- Gu, Goel, Ré. *S4: Efficiently Modeling Long Sequences with Structured State Spaces*, 2021.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
- Dao, Gu. *Transformers are SSMs* (Mamba-2), 2024.
- Mamba-3, 2026.
- Dao et al. *FlashAttention*, 2022. FlashAttention-2, 2023.
- Williams, Waterman, Patterson. *Roofline*, 2009.
- Sze, Chen, Yang, Emer. *Efficient Processing of Deep Neural Networks*, 2020.
- AI21 Jamba, 2024.

## Citation

```bibtex
@software{ssm_bench,
  author  = {Lincoln Angufibo},
  title   = {ssm-bench: reproducible benchmarks for state-space, attention and hybrid sequence models},
  year    = {2026},
  url     = {https://github.com/Continum-Computing/ssm-bench}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Continuum SSU-1

This repo measures the GPU gap. The [kalman](https://github.com/Continum-Computing/kalman) repo (private) designs the custom processor meant to close it. ssm-bench produces the data, kalman consumes it.
