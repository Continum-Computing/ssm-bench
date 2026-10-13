# GPU Benchmark Comparison: T4 vs A100

> **Correction, 2026-10-13.** The raw data in this document is unchanged. The interpretation is not. An earlier version concluded that the hypothesis was "strongly confirmed" and that custom silicon was justified. That conclusion does not follow from these measurements, because both sides of the comparison are naive reference implementations. The measured latencies are dominated by kernel launch overhead, not by how a GPU handles a well-implemented SSM. Everything below states what the data shows and what it does not.

## Executive Summary

We benchmarked reference implementations of SSM (recurrent and scan) and attention on Tesla T4 and A100-SXM4-40GB, 36 configurations per GPU.

The SSM path is a Python for-loop over time steps, so a 4096-step sequence issues thousands of small kernel launches. The attention path is a plain matmul-softmax-matmul. Neither uses fused kernels, and neither uses CUDA graphs.

The result: SSM appears 100-200x slower on T4 and 1,000-1,500x slower on A100, with both sides using a tiny fraction of the hardware. The gap is real as a measurement of our code. It is not evidence about GPU architecture, and it is not a go/no-go result.

## Hardware Specifications

| Metric | Tesla T4 | A100-SXM4-40GB | Ratio |
|--------|----------|----------------|-------|
| Compute Capability | 7.5 | 8.0 | - |
| Memory | 15.6 GB | 42.4 GB | 2.7x |
| Peak Bandwidth | 900 GB/s | 2039 GB/s | 2.3x |
| Peak FP32 | 13.0 TFLOPS | 19.5 TFLOPS | 1.5x |
| Peak BF16 | 26.0 TFLOPS | 78.0 TFLOPS | 3.0x |

## Performance Comparison

### Average Latency (all configurations)

| Model | T4 | A100 | A100 speedup | Ratio to attention |
|-------|----|----|---------|------------------|
| SSM Recurrent | 388.312 ms | 319.369 ms | 1.22x | T4 120x, A100 1,373x |
| SSM Scan | 497.913 ms | 413.016 ms | 1.21x | T4 153x, A100 1,776x |
| Attention | 3.247 ms | 0.233 ms | 13.96x | baseline |

The A100 is faster on every path, but not by the same amount. Attention gains 13.96x from the better GPU while the SSM path gains 1.22x.

That asymmetry is the honest form of what an earlier draft called "no speedup at all". The earlier draft was reading a corrupted T4 summary block that had been filled with the A100's averages, which made the two GPUs look identical at 319.4 ms. The T4's own raw data gives 388.312 ms. Each `metadata.json` now carries a `summary_repair` block listing every field whose stored value contradicted its raw data.

A launch-bound workload gaining 1.22x while a large-matmul workload gains 13.96x is what better hardware is expected to do, and it is not evidence about architecture. When per-step Python and launch overhead set the floor, extra bandwidth and FLOPs buy very little.

### Bandwidth Utilization

| Model | T4 | A100 |
|-------|----|----|
| SSM Recurrent | 0.062-0.415% | 0.015-0.106% |
| SSM Scan | 0.051-0.321% | 0.012-0.084% |
| Attention | 1.62-3.28% | 1.33-12.29% |

Peaks: T4 320 GB/s, A100-SXM4-40GB 1555 GB/s. An earlier version of this table divided by 900 and 2039 GB/s, which understated T4 utilization by 2.8x and A100 by 1.3x.

The SSM percentage is lower on A100 because the denominator grew, not because the workload got worse. Achieved GB/s for SSM is close on both GPUs (1.33 GB/s on T4, 1.53 GB/s on A100 at seq_len=4096), which points at a launch-bound floor rather than a bandwidth-bound one. Attention does move real data (10.0 GB/s to 189.6 GB/s) because a single large matmul amortizes its launch cost.

### Compute Utilization

| Model | T4 | A100 |
|-------|----|----|
| SSM Recurrent | 0.0149-0.049% | 0.0004-0.002% |
| SSM Scan | 0.0123-0.038% | 0.0004-0.001% |
| Attention | 9.38-29.74% | 0.97-14.06% |

Peaks: T4 8.1 TFLOP/s. Turing has no BF16 tensor core path, so a bf16 run there is bounded by the FP32 rate. A100 312 TFLOP/s of dense BF16 tensor throughput. An earlier version divided the A100 by 78.0 TFLOPS, a non-tensor rate, which is why it reported attention reaching 55.5%. Against the correct ceiling it reaches 14.1%.

The SSM path stays under 0.05% everywhere. Near-zero utilization means the GPU is idle between launches, not that SSMs inherently cannot use a GPU.

## Detailed Configuration Comparison

### seq_len=512, state_dim=64

| Model | T4 Latency | A100 Latency | T4 BW | A100 BW |
|-------|------------|--------------|-------|---------|
| SSM Recurrent | 109.6 ms | 86.0 ms | 0.20 GB/s (0.02%) | 0.25 GB/s (0.01%) |
| SSM Scan | 133.1 ms | 105.7 ms | 0.16 GB/s (0.02%) | 0.21 GB/s (0.01%) |
| Attention | 0.69 ms | 0.14 ms | 5.36 GB/s (0.60%) | 25.4 GB/s (1.25%) |

Gap: 160x on T4, 614x on A100

### seq_len=4096, state_dim=256

| Model | T4 Latency | A100 Latency | T4 BW | A100 BW |
|-------|------------|--------------|-------|---------|
| SSM Recurrent | 829.1 ms | 707.3 ms | 1.33 GB/s (0.15%) | 1.53 GB/s (0.08%) |
| SSM Scan | 1056.7 ms | 863.8 ms | 1.02 GB/s (0.11%) | 1.25 GB/s (0.06%) |
| Attention | 8.14 ms | 0.45 ms | 10.0 GB/s (1.11%) | 189.6 GB/s (9.30%) |

Gap: 102x on T4, 1,573x on A100

At seq_len=512 the SSM path takes 86-110ms for work that amounts to a few hundred small matrix-vector products. That is launch overhead, not arithmetic. It is the single clearest sign that this dataset measures the wrong thing.

## Arithmetic Intensity Analysis

| Model | Arithmetic Intensity | Bottleneck |
|-------|---------------------|------------|
| SSM Recurrent | 3-6 FLOP/byte | Low intensity |
| SSM Scan | 3-6 FLOP/byte | Low intensity |
| Attention | 146-230 FLOP/byte | High intensity |

This is the most durable result in the document. Arithmetic intensity follows from the algorithm, not from our implementation, so it survives the caveat that sinks the latency numbers.

SSMs do a handful of FLOPs per byte moved. Attention does hundreds. On a roofline, those sit in different regimes, and a machine tuned for the high-intensity regime will be a poorer fit for the low-intensity one. How poor that fit is in practice, against fused kernels, is what the strong baselines have to answer.

Note the labels above say "low intensity" and "high intensity" rather than "memory-bound" and "compute-bound". A workload cannot be diagnosed as bandwidth-bound while using 0.1% of available bandwidth. The intensity is a real property; the bottleneck diagnosis requires a measurement that actually saturates something.

## Scaling Behavior

### Latency vs Sequence Length

T4:
- SSM: 109ms to 829ms (7.6x for 8x seq_len)
- Attention: 0.69ms to 8.14ms (11.8x for 8x seq_len)

A100:
- SSM: 86ms to 707ms (8.2x for 8x seq_len)
- Attention: 0.14ms to 0.45ms (3.2x for 8x seq_len)

SSM latency grows linearly with sequence length, matching the O(N) recurrence. Attention grows superlinearly on T4 and sub-linearly on A100 at these sizes, which is what you expect while the whole computation still fits comfortably in cache and one launch dominates.

Sequence lengths of 512-4096 are below the range where attention's O(N²) term becomes the story. Testing at 32k is on the roadmap and matters for this comparison.

### Bandwidth Utilization vs Sequence Length

T4:
- SSM: 0.02% to 0.15%
- Attention: 0.60% to 1.15%

A100:
- SSM: 0.01% to 0.08%
- Attention: 1.25% to 9.30%

Longer sequences raise utilization for both, because there is more work per launch. Neither implementation approaches the bandwidth ceiling at any tested size.

## Conclusions

### 1. What the data supports

- **Arithmetic intensity differs by roughly 30-50x** between the SSM and attention paths. This is an algorithmic property and the strongest signal in the dataset.
- **The SSM recurrence is latency-bound at batch size 1** with a per-step Python loop. Thousands of small launches dominate.
- **Both implementations use a negligible share of the hardware.** Neither T4 nor A100 is close to saturated by anything we ran.
- **A100 helps attention and does not help our SSM path.** The reason is launch overhead, which better hardware does not fix.

### 2. What the data does not support

- **Not an architectural mismatch.** No fused SSM kernel was measured, so we have no evidence about how GPUs handle a properly implemented recurrence.
- **Not a go/no-go result.** The thresholds we set (40% / 60% headroom below roofline) were meant to be evaluated against strong baselines. Nothing here meets that bar.
- **Not a throughput-per-watt claim.** There is no power telemetry in the codebase. No energy was measured on either GPU.
- **Not a speedup target.** Earlier versions of this document cited "1000x improvement potential" and a "100-1000x speedup target". Those figures were derived from utilization percentages of a launch-bound workload and have no physical meaning. They are removed.

### 3. Root cause of the measured gap

The gap is dominated by implementation, in this order:

1. **Python loop over time steps** - interpreter overhead and one launch per step
2. **No kernel fusion** - each state update is several separate GPU operations
3. **Batch size 1** - no work to amortize fixed costs against
4. **No CUDA graphs** - launch cost paid in full, thousands of times

The algorithmic difference in arithmetic intensity is underneath all of this and is real, but at these sizes it is not what the latency table measures.

### 4. What has to be measured next

In priority order:

1. Official Mamba CUDA kernels (`mamba-ssm`), recurrent and scan paths
2. FlashAttention-2 at matching precision
3. A fused Mamba block (`mamba_ssm.modules.mamba_simple.Mamba`) for layer-level comparison
4. CUDA graph capture for decode, to remove launch overhead from the measurement
5. Energy per token via pynvml, since throughput-per-watt is the actual claim
6. Sequence lengths to 32k, where attention's quadratic term becomes visible
7. Batch sizes above 1, since serving workloads do not run at batch 1

Only after 1 through 5 does the go/no-go question become answerable.

### 5. Implications for the Kalman design work

Design work in the private `kalman` repo is blocked on the go/no-go decision that this dataset cannot produce. Code for the architecture, compiler, and RTL was written ahead of that gate on 2026-10-10 and is now frozen: not extended, not reviewed, not tested, and not counted as progress. Anything derived from these numbers is provisional and is discarded if the decision is no-go.

Two implications do hold, because they rest on arithmetic intensity rather than on our implementation:

- **State must live close to compute.** A 3-6 FLOP/byte workload cannot tolerate off-chip round trips per step.
- **Small matrix-vector operations, not large matmuls.** The compute units should be sized for the actual operation shape.

Everything else waits for the strong baseline data.

## Appendix: Raw Data

### T4 Results
- File: `t4-colab-2026-10-10/benchmark_results.json`
- Configurations: 36 (4 seq_lens × 3 state_dims × 3 models)
- Date: 2026-10-10
- Platform: Google Colab

### A100 Results
- File: `a100-modal-2026-10-10/benchmark_results.json`
- Configurations: 36 (4 seq_lens × 3 state_dims × 3 models)
- Date: 2026-10-10
- Platform: Modal (modal.com)

Raw JSON files are immutable. This analysis is interpretation and is corrected in place, with the correction dated at the top.

---

**Bottom line:** the GPU-SSM mismatch is a reasonable hypothesis with a real algorithmic basis in arithmetic intensity. This dataset does not test it. The strong baseline work will.
