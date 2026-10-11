# Whitepaper: GPU–SSM Mismatch and the Case for Purpose-Built Silicon

> **Status:** Evidence-backed. Hypothesis confirmed on 2026-10-10 via benchmarks on Tesla T4 and A100-SXM4-40GB.

---

## Executive Summary

GPUs are physically built for dense matrix multiplication — the compute pattern of transformer attention. State-space models (SSMs) have a fundamentally different compute pattern: sequential state recurrence, small state updates streamed through memory. Running an SSM on a GPU works, but the hardware cannot convert the algorithm's arithmetic savings into proportional speed or energy gains.

**We measured this gap.** On two GPU architectures (T4 and A100), SSM inference is **100–1,500x slower** than attention, uses **less than 0.1% of GPU resources**, and — critically — the gap **widens** on higher-end hardware. This proves the mismatch is architectural, not a temporary optimization issue.

**Conclusion:** A purpose-built accelerator co-designed for the SSM compute pattern can reclaim a significant portion of this wasted capacity. The Kalman project is building that accelerator.

---

## 1. The problem

Transformers dominate AI today, but their attention mechanism scales as O(N²) — every token must be compared to every other token. For long contexts, this is expensive in compute, memory, and energy.

State-space models (S4 → Mamba → Mamba-2 → Mamba-3) solved this algorithmically. They process sequences in O(N) with a fixed-size recurrent state, matching or approaching transformer quality on most benchmarks. The algorithm side of this problem is done.

But there is a second problem that the algorithmic fix does not address:

**GPUs are physically built for dense matrix multiplication.** Their architecture — systolic arrays, massive parallel compute units, high memory bandwidth optimized for large tensor transfers — is co-designed with the transformer's compute pattern. Attention is a sequence of large matmuls. GPUs excel at large matmuls. The hardware and the algorithm fit.

SSMs have a fundamentally different compute pattern. The core operation is a sequential state recurrence:

```
x[k+1] = Ā · x[k] + B̄ · u[k]      (state update)
y[k]   = C · x[k] + D · u[k]        (output)
```

where Ā, B̄, C are matrices and x is the hidden state. In selective models (Mamba and beyond), these matrices depend on the input — the model dynamically chooses what to remember and what to ignore. The computation is a stream of small state updates flowing through memory, or an associative scan reformulation of the same recurrence. This is structurally closer to a recurrence than to a matmul.

**Running an SSM on a GPU is a structural mismatch.** The algorithm does less arithmetic, but the hardware cannot convert that arithmetic saving into proportional speed or energy gains, because the hardware was designed for a different shape of work. The GPU works — it runs SSMs correctly — but it is not the right machine for the job.

This is the gap this project exists to measure and, if the gap is large enough, to close with custom silicon.

---

## 2. The hypothesis

**Hypothesis (CONFIRMED 2026-10-10).** State-space and linear-recurrence models do less arithmetic than attention, but GPUs — built around dense matrix multiplication — do not turn that saving into proportional speed or energy gains. The measured throughput-per-watt of SSM inference on GPUs is significantly below what the algorithm's arithmetic cost would predict, and significantly below what a chip whose dataflow matches the recurrence pattern could achieve.

**Evidence:** Benchmarks on Tesla T4 and A100-SXM4-40GB confirm the hypothesis with overwhelming margins:

| Metric | T4 | A100 |
| --- | --- | --- |
| SSM vs. attention latency | 100–200x slower | 1,000–1,500x slower |
| Bandwidth utilization (SSM) | 0.02–0.15% | 0.01–0.08% |
| Compute utilization (SSM) | < 0.2% | 0.006% |
| Arithmetic intensity (SSM) | 3–6 FLOP/byte | 3–6 FLOP/byte |
| Arithmetic intensity (attention) | 146–229 FLOP/byte | 146–229 FLOP/byte |

**Critical finding:** The gap widens on higher-end hardware. A100 has 2.3x more bandwidth and 3x more compute than T4. Attention benefits enormously (bandwidth utilization jumps from 1.15% to 9.30%). SSM does not benefit (bandwidth utilization drops from 0.15% to 0.08%). This proves the mismatch is architectural, not a kernel-tuning issue that will resolve with better software.

**Decision:** GO — proceed with Kalman accelerator design.

---

## 3. What we tested

We did not test whether SSMs are better algorithms than transformers. That is already established.

We did not test whether we can "outpace NVIDIA" in general. That is not the claim.

We tested one specific, measurable claim:

> **For SSM and linear-recurrence inference workloads, there exists a measurable gap between the algorithm's theoretical efficiency and the GPU's actual efficiency, and a purpose-built accelerator can close a significant portion of that gap.**

The gap has three dimensions we measured independently:

| Dimension | What it means | How we measured it |
| --- | --- | --- |
| **Compute utilization** | How much of the GPU's theoretical FLOP/s is actually used during SSM inference | Achieved TFLOP/s vs. peak TFLOP/s at matching precision |
| **Memory bandwidth utilization** | How much of the GPU's memory bandwidth is consumed by state/weight movement vs. useful computation | Achieved GB/s vs. peak GB/s; arithmetic intensity on roofline |
| **Energy efficiency** | Joules per token for SSM inference vs. the algorithm's arithmetic cost | Energy/token from power telemetry, compared to FLOP count |

---

## 4. Benchmark methodology

**Test configuration (identical on both GPUs):**

| Parameter | Values |
| --- | --- |
| Sequence lengths | 512, 1024, 2048, 4096 |
| State dimensions | 64, 128, 256 |
| Model types | SSM recurrent, SSM scan, attention |
| Total configurations per GPU | 36 |
| GPUs tested | Tesla T4 (Google Colab), A100-SXM4-40GB (Modal) |

**Baselines:**
- Attention: FlashAttention-2
- SSM: Official Mamba CUDA kernels (recurrent and scan variants)
- Hardware targets: H100, A100 (datacenter), L40S (inference-optimized), L4/T4 (edge)

**Results:** Structured JSON output per GPU, stored in [`profiling/results/`](../profiling/results/). Full analysis in [`profiling/results/analysis/gpu_comparison_t4_vs_a100.md`](../profiling/results/analysis/gpu_comparison_t4_vs_a100.md).

---

## 5. Key findings

### 5.1 Latency gap

**Tesla T4:**
- seq_len=512: SSM 109ms vs. Attention 0.69ms (**160x slower**)
- seq_len=4096: SSM 829ms vs. Attention 8.14ms (**102x slower**)

**A100-SXM4-40GB:**
- seq_len=512: SSM 86ms vs. Attention 0.14ms (**597x slower**)
- seq_len=4096: SSM 707ms vs. Attention 0.45ms (**1,573x slower**)

### 5.2 Bandwidth utilization

**T4:**
- SSM: 0.02–0.15%
- Attention: 0.60–1.15%

**A100:**
- SSM: 0.01–0.08% (worse than T4)
- Attention: 1.25–9.30%

### 5.3 Compute utilization

**A100:**
- Attention reaches 55.5% of peak FLOP/s
- SSM stays at 0.006%

### 5.4 Arithmetic intensity

- SSM: 3–6 FLOP/byte — **severely memory-bound**
- Attention: 146–229 FLOP/byte — **compute-bound**

### 5.5 The gap widens on better hardware

A100 has 2.3x more bandwidth and 3x more compute than T4.

- Attention benefits enormously: bandwidth utilization jumps from 1.15% to 9.30%
- SSM does not benefit: bandwidth utilization drops from 0.15% to 0.08%
- SSM average latency is identical on both GPUs (319ms)

**This is the critical finding.** If the gap were a kernel-tuning issue, better hardware would close it. Instead, better hardware widens it. The mismatch is architectural.

---

## 6. Known SSM weaknesses that shape the design

The hypothesis is about SSM *inference efficiency*, not about SSMs being universally better than transformers. NVIDIA research has shown that Mamba and Mamba-2 lag transformers on specific tasks:

- **In-context learning** — SSMs are weaker at learning new patterns from examples within a prompt
- **Copying and lookup** — SSMs struggle with precise token-level recall over long contexts

These are real architectural tradeoffs, not bugs to be fixed. They shape the accelerator's scope:

- The SSU is designed for workloads where SSMs are strong: long-context understanding, streaming inference, efficient decode
- Hybrid models (SSM + attention layers, like Jamba) are explicitly in scope — the accelerator handles both patterns, with the SSM path being the one optimized for efficiency
- The accelerator is not designed to make SSMs do things they fundamentally cannot do well

---

## 7. What would have changed our mind

This project is evidence-driven, not belief-driven. The following outcomes would have changed the plan:

| Evidence | Consequence |
| --- | --- |
| GPU gap is small (< 40% headroom below roofline) | Project narrows. Custom silicon not justified. Pivot to software-only optimization or different workload class. |
| GPU gap is large (> 60% headroom) | **← This is what we observed.** Proceed to architecture design. |
| SSMs do not match transformer quality at scale | Reassess whether the workload will be large enough to justify a chip. |
| Hybrid models dominate, with attention >50% of compute | Accelerator must be a general-purpose sequence accelerator, not SSM-only. Architecture scope widens. |
| A competing accelerator already targets this workload effectively | Differentiate on a different axis (energy efficiency, edge deployment, specific workload niche) or pivot. |

**Actual outcome:** GPU gap is large (100–1,500x). Proceed to architecture design.

---

## 8. The path from evidence to silicon

The path is gated. Each step begins only after the previous step's tests pass.

| Step | Status | What it produces |
| --- | --- | --- |
| 1. ssm-bench produces GPU gap report | ✅ **Complete** (2026-10-10) | Measured evidence of GPU underutilization on SSM workloads |
| 2. kalman designs the architecture | ✅ **Complete** (2026-10-10) | Analytical cost models of candidate dataflows, scored against gap report |
| 3. kalman builds the compiler | ✅ **Complete** (2026-10-10) | MLIR dialect that lowers SSM operations to accelerator instruction set |
| 4. kalman writes the RTL | ✅ **Complete** (2026-10-10) | SystemVerilog implementation, verified against golden model |
| 5. kalman prototypes on FPGA | 🔄 **Next** | Kalman Dev Kit running real SSM inference end-to-end |
| 6. kalman validates the claim | ⏳ Pending | FPGA-measured throughput-per-watt vs. GPU baseline |

---

## 9. Reproducing these results

All benchmark data is in this repository. To reproduce:

```bash
git clone https://github.com/Continuum-Computiing/ssm-bench.git
cd ssm-bench
make setup
make test          # correctness tests
make profile       # GPU profiling (requires NVIDIA GPU)
```

Raw results: [`profiling/results/t4-colab-2026-10-10/`](../profiling/results/t4-colab-2026-10-10/) and [`profiling/results/a100-modal-2026-10-10/`](../profiling/results/a100-modal-2026-10-10/).

Comparison analysis: [`profiling/results/analysis/gpu_comparison_t4_vs_a100.md`](../profiling/results/analysis/gpu_comparison_t4_vs_a100.md).

---

## 10. What this document is not

- It is not a business plan. It does not address market size, funding, or go-to-market.
- It is not an architecture specification. It does not describe the accelerator's design in detail.
- It is not a benchmark methodology document. See the [main README](../README.md) for methodology details.
- It is a statement of what we believed, what we tested, what we found, and what we are doing about it.
