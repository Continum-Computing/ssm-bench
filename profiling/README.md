# GPU Profiling Infrastructure

This directory contains tools for measuring and analyzing GPU performance of state-space models (SSMs) vs attention mechanisms. The goal is to quantify the "GPU-SSM mismatch" — whether GPUs underutilize their capabilities on SSM workloads compared to attention.

## Overview

The profiling infrastructure measures:

1. **Latency** — Time per forward pass (ms)
2. **Memory Bandwidth** — Bytes transferred per second (GB/s) and utilization vs theoretical peak
3. **Compute Performance** — FLOPS achieved and utilization vs theoretical peak
4. **Arithmetic Intensity** — FLOPS per byte transferred (identifies memory-bound vs compute-bound workloads)
5. **Roofline Analysis** — Visualizes where each workload sits relative to hardware limits

## Quick Start

### Run Benchmarks

```bash
# Run default benchmark sweep (512, 1024, 2048, 4096 sequence lengths)
python profiling/run_benchmarks.py

# Custom configuration
python profiling/run_benchmarks.py \
  --seq-lens 256 512 1024 2048 4096 8192 \
  --state-dims 64 128 256 512 \
  --input-dim 512 \
  --output-dim 512 \
  --dtype bf16 \
  --device 0 \
  --output profiling/results/my_benchmark.json
```

### Generate Visualizations

```bash
# After running benchmarks, generate all plots and reports
python profiling/visualize.py profiling/results/benchmark_results.json
```

This generates:
- `latency_vs_seq_len.png` — Latency scaling with sequence length
- `bandwidth_utilization.png` — Memory bandwidth utilization
- `compute_utilization.png` — Compute utilization
- `roofline_model.png` — Roofline analysis
- `benchmark_summary.txt` — Text summary report

## What We Measure

### SSM (State-Space Model)

We benchmark two SSM implementations:
- **Recurrent** — Sequential processing, O(N) time, O(1) memory
- **Scan** — Parallel scan algorithm, O(N) time, O(N) memory (but parallelizable)

Both use the same mathematical formulation:
```
x[k+1] = Ā_k · x[k] + B̄_k · u[k]  (state update)
y[k]   = C · x[k] + D · u[k]        (output)
```

Where Ā_k and B̄_k are discretized matrices that depend on the input (selective mechanism).

### Attention

We benchmark standard multi-head attention:
```
Q = X · W_Q
K = X · W_K
V = X · W_V
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
Output = Attention · W_O
```

### Metrics

**Latency (ms)**
- Mean, median, P95, P99 over 100 runs
- Warmup: 10 runs (discarded)

**Memory Bandwidth (GB/s)**
- Calculated from total bytes transferred / time
- Includes: input, output, weights, intermediate activations
- Utilization = achieved / theoretical peak × 100%

**Compute (TFLOPS)**
- Calculated from total FLOPS / time
- FLOPS counted: matrix multiplications, element-wise operations
- Utilization = achieved / theoretical peak × 100%

**Arithmetic Intensity (FLOP/Byte)**
- Ratio of compute to memory traffic
- Low intensity (< 10) = memory-bound
- High intensity (> 100) = compute-bound

## Interpreting Results

### The GPU-SSM Mismatch Hypothesis

We expect to find:

1. **SSM has lower arithmetic intensity than attention**
   - SSM: Many small matrix-vector operations (state updates)
   - Attention: Large matrix-matrix multiplications (Q·K^T)
   - GPUs are optimized for large matmuls, not small matvecs

2. **SSM achieves lower bandwidth utilization**
   - SSM's sequential nature limits memory access patterns
   - Attention's parallel structure enables better memory coalescing
   - Result: SSM leaves bandwidth on the table

3. **SSM achieves lower compute utilization**
   - SSM's small operations don't saturate GPU compute units
   - Attention's large operations fully utilize tensor cores
   - Result: SSM leaves FLOPS on the table

### What the Data Tells Us

**If SSM bandwidth utilization < 50%:**
- Strong evidence for custom accelerator
- GPUs are not optimized for SSM workloads
- Custom dataflow could reclaim significant bandwidth

**If SSM compute utilization < 30%:**
- Strong evidence for custom accelerator
- GPU compute units are underutilized
- Custom compute units could be more efficient

**If SSM latency is 2-5x worse than attention:**
- Confirms the mismatch is significant
- Justifies hardware investment

**If SSM performs similarly to attention:**
- Hypothesis may be wrong
- GPUs may already handle SSMs well
- Custom accelerator may not provide significant benefits

## Configuration Options

### BenchmarkConfig

```python
BenchmarkConfig(
    seq_len=1024,        # Sequence length
    batch_size=1,        # Batch size (currently fixed at 1)
    state_dim=128,       # SSM state dimension
    input_dim=256,       # Input dimension
    output_dim=256,      # Output dimension
    dtype="bf16",        # Data type: "fp32", "fp16", "bf16"
)
```

### Command-Line Arguments

```bash
--seq-lens      # List of sequence lengths to test
--state-dims    # List of SSM state dimensions to test
--input-dim     # Input dimension (fixed for all runs)
--output-dim    # Output dimension (fixed for all runs)
--dtype         # Data type (fp32, fp16, bf16)
--device        # CUDA device ID (default: 0)
--output        # Output JSON path
```

## File Structure

```
profiling/
├── __init__.py              # Package exports
├── gpu_profiler.py          # Core profiling utilities
├── benchmark_runner.py      # Benchmark orchestration
├── run_benchmarks.py        # Main entry point
├── visualize.py             # Visualization and reporting
└── results/                 # Benchmark output directory
    ├── benchmark_results.json
    ├── latency_vs_seq_len.png
    ├── bandwidth_utilization.png
    ├── compute_utilization.png
    ├── roofline_model.png
    └── benchmark_summary.txt
```

## Requirements

- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- matplotlib (for visualization)
- numpy

Install dependencies:
```bash
pip install torch matplotlib numpy
```

## Example Output

### Benchmark Summary

```
GPU: NVIDIA A100-SXM4-80GB
Compute Capability: 8.0
Total Memory: 80.0 GB
Peak Memory Bandwidth: 2039 GB/s
Peak FP32 Performance: 19.5 TFLOPS
Peak BF16 Performance: 78.0 TFLOPS

Average Latency (ms):
  SSM Recurrent: 12.345 ms
  SSM Scan:      8.234 ms
  Attention:     15.678 ms

Average Bandwidth Utilization (%):
  SSM Recurrent: 23.4%
  SSM Scan:      31.2%
  Attention:     67.8%

Average Compute Utilization (%):
  SSM Recurrent: 8.7%
  SSM Scan:      12.3%
  Attention:     45.6%

SSM Speedup vs Attention:
  Recurrent: 1.27x faster
  Scan:      1.90x faster

GPU Utilization Gap:
  SSM Recurrent bandwidth utilization: 23.4%
  Attention bandwidth utilization:     67.8%
  Gap: 44.4 percentage points

Interpretation:
  ✓ SSM shows significantly lower bandwidth utilization than attention,
    suggesting GPUs are not optimized for SSM workloads.
    This supports the hypothesis for custom SSM accelerator design.
```

## Next Steps

After running benchmarks and analyzing results:

1. **If hypothesis confirmed** (SSM utilization < 50%):
   - Proceed with Kalman accelerator design
   - Use profiling data to guide architecture decisions
   - Focus on improving bandwidth utilization for SSM workloads

2. **If hypothesis not confirmed** (SSM utilization > 70%):
   - Re-evaluate the need for custom accelerator
   - Consider software optimizations instead
   - Pivot to different workload or approach

3. **Document findings**:
   - Update `docs/gap-report/` with detailed analysis
   - Create decision record in kalman repo
   - Share results with team

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
- Reduce `--seq-lens` (start with smaller values)
- Reduce `--state-dims`
- Use `--dtype fp16` or `--dtype bf16` instead of `fp32`

### Slow Benchmark Runs

Benchmarks can take time, especially for large configurations:
- Default sweep: 4 seq_lens × 3 state_dims × 3 models = 36 configurations
- Each configuration: 10 warmup + 100 benchmark runs
- Estimated time: 5-15 minutes depending on GPU

### No GPU Available

If you see "CUDA not available":
- Ensure NVIDIA drivers are installed
- Ensure PyTorch is installed with CUDA support
- Check `nvidia-smi` shows your GPU

## References

- [Roofline Model (Williams et al., 2009)](https://doi.org/10.1145/1498765.1498785)
- [Mamba Paper (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
