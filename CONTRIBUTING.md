# Contributing to ssm-bench

ssm-bench measures how efficiently GPUs run state-space models, attention, and hybrid sequence models. Everything here is public, reproducible, and used to make a real engineering decision, so contributions are held to a numerical standard rather than a stylistic one.

## Where the project actually stands

Read this before opening a PR, because it determines what is useful.

The published results compare our own **reference implementations**: a Python for-loop SSM against a plain matmul-softmax-matmul attention, at batch size 1, in bf16. Neither side uses fused kernels. Those numbers measure kernel launch overhead far more than they measure architecture, and we say so in the README, the whitepaper, and the analysis document.

The project has a go/no-go gate: no downstream hardware work proceeds until the gap has been measured against **strong baselines**. That measurement does not exist yet. Issues labelled `gate` are the work that produces it, and they are the highest-value contribution to this repository.

If you want to help and are not sure where to start, look at the `gate` label first.

## What makes a benchmark contribution acceptable

A baseline or measurement PR is merged when all of the following hold. These are not formalities; every one of them exists because it was gotten wrong once already.

**1. Correctness before timing.** Any new implementation is checked against the NumPy reference in `model/reference/` before it is ever timed. A fast kernel that computes the wrong thing is a bug report, not a result.

**2. Peaks come from the spec table, never from a literal.** Utilization percentages are only as good as the denominator. Every peak must come from `profiling/gpu_specs_db.py`, looked up by device name. Do not hardcode a bandwidth or TFLOP/s figure in a benchmark script. If the GPU is not in the table, add it there with a source, and note that an unrecognised GPU raises deliberately rather than falling back to a guess.

Two specific traps:

- Compute capability is not specific enough. Tesla T4 and RTX 2080 Ti are both capability 7.5 and differ by roughly 3x in bandwidth. An earlier version of this repo keyed on capability and silently scored a T4 against 2080 Ti numbers.
- Use **dense** tensor-core rates, not sparse. Sparse figures assume 2:4 structured sparsity that these workloads do not use, and will understate utilization by about 2x.
- Match the precision actually run. Comparing bf16 throughput against an FP32 ceiling, or against a non-tensor rate, is meaningless. Turing and Volta have no bf16 tensor-core path, so a bf16 run there is bounded by the FP32 rate.

**3. Warm up, then report a distribution.** Warm-up iterations are discarded. Report median and P95 over at least 100 runs, never a single run and never a bare mean.

**4. Results go in the right place, with metadata.** New runs live in `profiling/results/{gpu}-{platform}-{YYYY-MM-DD}/` containing `benchmark_results.json`, `metadata.json`, and plots where generated. Update `profiling/results/manifest.json`. The schema is in [`docs/results-schema.md`](docs/results-schema.md).

**5. Measurements are immutable; interpretations are not.** Once committed, `latency_stats`, `flops`, `bytes_transferred`, `achieved_bandwidth_gb_s`, `achieved_tflops`, and `arithmetic_intensity` are never edited. If a measurement is wrong, a new corrected run is added. Derived values and prose may be corrected, with a dated note explaining what changed and why.

**6. State the denominator.** Any utilization figure in documentation names the peak and the precision it was divided by. "0.4% of bandwidth" is not a claim; "0.4% of the T4's 320 GB/s peak" is.

**7. Do not overstate what a run shows.** If both sides of a comparison are naive implementations, say so in the result itself and not only in a caveat elsewhere. If a metric was not measured, it is not reported. Energy per token is specified in the README but unimplemented, so no throughput-per-watt figure appears anywhere in this repo.

## Adding a GPU

Add an entry to `GPU_PEAK_SPECS` in `profiling/gpu_specs_db.py`, most specific name first, since matching is by substring against the CUDA device name. Include bandwidth, FP32, FP16 tensor, and BF16 rates, plus a `note` recording the source and any caveat. Then run the sweep and publish the result set.

## Running things

```bash
make setup     # virtual environment and dependencies
make test      # correctness tests, CPU only
make lint
python profiling/run_benchmarks.py    # needs an NVIDIA GPU
python profiling/visualize.py profiling/results/<run>/benchmark_results.json
```

CPU-only machines can run the correctness suite and the analysis tooling. GPU runs need Colab, Modal, or hardware you own; `model/modal_benchmark.py` is a worked example for Modal.

## Style

- Write like a person, not a press release. No em-dashes, no marketing cadence.
- Say "custom processor", not "accelerator".
- Numbers carry units and a denominator. Ratios say which direction they run: "SSM is 120x slower than attention", never "120x speedup" when the value is below one.
- A claim that cannot be reproduced from a script and a metadata file does not belong in the docs.

## Commits and pull requests

Commit messages are `type: summary`, where type is one of `feat`, `fix`, `docs`, `test`, `results`, `chore`.

One logical change per branch. In the PR description, say what was measured, on what hardware, against which peaks, and what the numbers do and do not support. If a PR changes any published figure, explain why the old figure was wrong.

No secrets, no credentials, no generated artefacts outside `profiling/results/`.

## Licensing

Contributions are made under the Apache 2.0 licence in [LICENSE](LICENSE). Confirm you have the right to contribute and that no employer or contract term conflicts.
