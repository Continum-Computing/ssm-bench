# GPU Profiling Infrastructure

Tools for measuring and analyzing GPU performance of state-space models (SSMs) vs. attention mechanisms.

## Overview

The profiling infrastructure measures where each workload sits relative to hardware limits:

1. **Latency** - time per forward pass (ms)
2. **Memory bandwidth** - bytes transferred per second (GB/s) and utilization vs. theoretical peak
3. **Compute performance** - FLOPS achieved and utilization vs. theoretical peak
4. **Arithmetic intensity** - FLOPS per byte transferred (identifies memory-bound vs. compute-bound workloads)
5. **Roofline analysis** - visualizes where each workload sits relative to hardware limits

What is being profiled is this repo's own reference code: a Python for-loop SSM and a plain matmul attention, both at batch size 1. Read every number with that in mind. Energy per token is not measured yet (there is no power telemetry code), and no fused kernels (Mamba CUDA, FlashAttention-2) are integrated.

## Quick Start

### Run benchmarks

```bash
# Default sweep (512, 1024, 2048, 4096 sequence lengths)
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

### Generate visualizations

```bash
python profiling/visualize.py profiling/results/benchmark_results.json
```

Generates:
- `latency_vs_seq_len.png` - latency scaling with sequence length
- `bandwidth_utilization.png` - memory bandwidth utilization
- `compute_utilization.png` - compute utilization
- `roofline_model.png` - roofline analysis
- `benchmark_summary.txt` - text summary report

## Implementations

### SSM (State-Space Model)

Two implementations:

- **Recurrent** - sequential processing, O(N) time, O(1) memory
- **Scan** - parallel scan algorithm, O(N) time, O(N) memory (parallelizable)

Both use the selective SSM formulation:
```
x[k+1] = Ā_k · x[k] + B̄_k · u[k]  (state update)
y[k]   = C · x[k] + D · u[k]        (output)
```

Where Ā_k and B̄_k are discretized matrices that depend on the input (selective mechanism).

### Attention

Standard multi-head attention:
```
Q = X · W_Q
K = X · W_K
V = X · W_V
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
Output = Attention · W_O
```

## Metrics

### Latency (ms)
- Mean, median, P95, P99 over 100 runs
- Warmup: 10 runs (discarded)

### Memory bandwidth (GB/s)
- Calculated from total bytes transferred / time
- Includes: input, output, weights, intermediate activations
- Utilization = achieved / theoretical peak × 100%

### Compute (TFLOPS)
- Calculated from total FLOPS / time
- FLOPS counted: matrix multiplications, element-wise operations
- Utilization = achieved / theoretical peak × 100%

### Arithmetic intensity (FLOP/byte)
- Ratio of compute to memory traffic
- Low intensity (< 10) = memory-bound
- High intensity (> 100) = compute-bound

## Configuration

### BenchmarkConfig

```python
BenchmarkConfig(
    seq_len=1024,  # Sequence length
    batch_size=1,  # Batch size (currently fixed at 1)
    state_dim=128,  # SSM state dimension
    input_dim=256,  # Input dimension
    output_dim=256,  # Output dimension
    dtype="bf16",  # Data type: "fp32", "fp16", "bf16"
)
```

### Command-line arguments

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

## Troubleshooting

### CUDA out of memory

- Reduce `--seq-lens` (start with smaller values)
- Reduce `--state-dims`
- Use `--dtype bf16` instead of `fp32`

### Slow benchmark runs

Default sweep: 4 seq_lens × 3 state_dims × 3 models = 36 configurations. Each configuration runs 10 warmup + 100 benchmark iterations. Estimated time: 5–15 minutes depending on GPU.

### No GPU available

- Ensure NVIDIA drivers are installed
- Ensure PyTorch is installed with CUDA support
- Check `nvidia-smi` shows your GPU

## References

- [Roofline Model (Williams et al., 2009)](https://doi.org/10.1145/1498765.1498785)
- [Mamba Paper (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
