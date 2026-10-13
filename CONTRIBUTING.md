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

These are the commands CI runs, and they work on any platform:

```bash
python -m pip install -e ".[dev]"   # dependencies, including the pinned ruff
python -m pytest                    # correctness tests, CPU only
python -m ruff check .              # lint
python -m ruff format --check .     # formatting
```

`make setup`, `make test` and `make lint` also exist, but the Makefile hardcodes a POSIX virtualenv layout (`$(VENV)/bin`) and fails on Windows. Until #36 settles that, prefer the direct commands above, which is what CI uses.

`python profiling/visualize.py <results.json>` writes into the directory holding that JSON: four plots under `plots/` and one `benchmark_summary.txt` beside it. Pass `--run-dir` to write elsewhere. It needs no GPU, since everything is recomputed from stored data.

`python profiling/run_benchmarks.py` needs an NVIDIA GPU.

The visualize command used to exit 0 and write nothing, because the module had no entry point and never imported argparse. Worth knowing, because a silent no-op is worse than a crash: you walk away believing the plots now exist. That is how the A100 result set came to have no artifacts, and a test now runs the script as a subprocess so the entry point cannot vanish quietly again.

CPU-only machines can run the correctness suite and the analysis tooling. GPU runs need Colab, Modal, or hardware you own; `model/modal_benchmark.py` is a worked example for Modal.

## Continuous integration

Every pull request and every push to `main` runs `.github/workflows/ci.yml`. Three checks report: `Lint`, `Test (Python 3.11)` and `Test (Python 3.12)`.

Lint runs ruff 0.17.0. That version is pinned in two places, `pyproject.toml` and `ci.yml`, and both must be bumped together. Install with `pip install -e ".[dev]"` and your local run matches CI. A different ruff version can disagree with the gate, because ruff retunes rules between releases.

One thing catches people out. `ruff format` also rewrites fenced `python` blocks inside Markdown, so editing a code example in a README can fail the lint job without touching a single `.py` file. Run `python -m ruff format .` before pushing and it will normalise the example for you.

Tests run against CPU-only PyTorch, because GitHub-hosted runners have no NVIDIA GPU. Nothing in CI validates a performance claim, and no CI job currently reads `profiling/results/`. A green suite means the reference implementations agree with each other. It does not mean the published numbers are right. That gap is #27.

`main` is protected. Changes arrive by pull request, and direct pushes are refused for maintainers too.

## Style

- Write like a person, not a press release. No em-dashes, no marketing cadence.
- Say "custom processor", not "accelerator".
- Numbers carry units and a denominator. Ratios say which direction they run: "SSM is 120x slower than attention", never "120x speedup" when the value is below one.
- A claim that cannot be reproduced from a script and a metadata file does not belong in the docs.
- Leave `A`, `B`, `C`, `D`, `A_bar` and `B_bar` in capitals. `N803` and `N806` are disabled in `pyproject.toml` precisely so those names keep the spelling they have in the state-space literature. Lowercasing them makes the reference implementations harder to read against the papers.

## Commits and pull requests

Commit messages are `type: summary`, where type is one of `feat`, `fix`, `docs`, `test`, `results`, `chore`.

One logical change per branch. In the PR description, say what was measured, on what hardware, against which peaks, and what the numbers do and do not support. If a PR changes any published figure, explain why the old figure was wrong.

No secrets, no credentials, no generated artefacts outside `profiling/results/`.

## Licensing

Contributions are made under the Apache 2.0 licence in [LICENSE](LICENSE). Confirm you have the right to contribute and that no employer or contract term conflicts.
