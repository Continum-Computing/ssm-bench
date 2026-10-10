"""
GPU Profiler - Core utilities for measuring GPU performance.

This module provides tools for:
- Querying GPU specifications and capabilities
- Precise timing using CUDA events
- Calculating memory bandwidth utilization
- Calculating compute utilization (FLOPS)
- Building roofline models

All measurements are designed to be accurate and reproducible.
"""

import torch
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class GPUSpecs:
    """GPU hardware specifications."""
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    memory_bandwidth_gb_s: float  # Theoretical peak
    fp32_tflops: float  # Theoretical peak
    fp16_tflops: float  # Theoretical peak
    bf16_tflops: float  # Theoretical peak
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "total_memory_gb": self.total_memory_gb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "fp32_tflops": self.fp32_tflops,
            "fp16_tflops": self.fp16_tflops,
            "bf16_tflops": self.bf16_tflops,
        }


def get_gpu_specs(device_id: int = 0) -> Optional[GPUSpecs]:
    """
    Query GPU specifications.
    
    Args:
        device_id: CUDA device ID (default: 0)
    
    Returns:
        GPUSpecs object or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None
    
    device = torch.device(f"cuda:{device_id}")
    props = torch.cuda.get_device_properties(device)
    
    # Get theoretical peak performance
    # These are approximate values for common GPUs
    # In production, you'd want a more comprehensive database
    compute_cap = (props.major, props.minor)
    
    # Estimate peak performance based on GPU architecture
    # These are rough estimates - actual values vary by specific model
    gpu_specs_db = {
        # (major, minor): (mem_bw_gb_s, fp32_tflops, fp16_tflops, bf16_tflops)
        (8, 0): (2039, 19.5, 78.0, 78.0),  # A100
        (8, 6): (600, 10.0, 40.0, 40.0),   # RTX 3090
        (8, 9): (768, 82.6, 165.2, 165.2), # H100
        (7, 5): (900, 13.0, 26.0, 26.0),   # RTX 2080 Ti
        (7, 0): (900, 14.0, 28.0, 28.0),   # V100
    }
    
    # Default to conservative estimates if GPU not in database
    mem_bw, fp32, fp16, bf16 = gpu_specs_db.get(
        compute_cap, 
        (300, 5.0, 10.0, 10.0)  # Conservative default
    )
    
    return GPUSpecs(
        name=props.name,
        compute_capability=compute_cap,
        total_memory_gb=props.total_memory / 1e9,
        memory_bandwidth_gb_s=mem_bw,
        fp32_tflops=fp32,
        fp16_tflops=fp16,
        bf16_tflops=bf16,
    )


class CUDATimer:
    """
    High-precision timer using CUDA events.
    
    CUDA events provide microsecond-precision timing that's synchronized
    with GPU execution, avoiding CPU-GPU synchronization overhead.
    """
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        """Start timing."""
        self.start_event.record()
    
    def stop(self) -> float:
        """
        Stop timing and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def benchmark_function(
    func,
    *args,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark a function with warmup and statistical analysis.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments to pass to function
        warmup_runs: Number of warmup iterations (not measured)
        benchmark_runs: Number of benchmark iterations (measured)
        **kwargs: Keyword arguments to pass to function
    
    Returns:
        Dictionary with timing statistics (mean, std, min, max, percentiles)
    """
    # Warmup phase
    for _ in range(warmup_runs):
        _ = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark phase
    times_ms = []
    
    if torch.cuda.is_available():
        # Use CUDA events for GPU timing
        timer = CUDATimer()
        
        for _ in range(benchmark_runs):
            timer.start()
            _ = func(*args, **kwargs)
            elapsed = timer.stop()
            times_ms.append(elapsed)
    else:
        # Fallback to CPU timing
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = func(*args, **kwargs)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    times_ms = sorted(times_ms)
    
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms))**0.5,
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "median_ms": times_ms[len(times_ms)//2],
        "p95_ms": times_ms[int(len(times_ms) * 0.95)],
        "p99_ms": times_ms[int(len(times_ms) * 0.99)],
        "runs": benchmark_runs,
    }


def calculate_flops_ssm(
    seq_len: int,
    state_dim: int,
    input_dim: int,
    output_dim: int,
) -> int:
    """
    Calculate theoretical FLOPS for selective SSM forward pass.
    
    Args:
        seq_len: Sequence length
        state_dim: SSM state dimension
        input_dim: Input dimension
        output_dim: Output dimension
    
    Returns:
        Total FLOPS for forward pass
    """
    # Discretization: A_bar = I + A * delta, B_bar = B * delta
    # For each timestep: state_dim^2 (A*delta) + state_dim*input_dim (B*delta)
    flops_per_step = state_dim**2 + state_dim * input_dim
    
    # State update: x = A_bar @ x + B_bar @ u
    # Matrix-vector multiply: 2 * state_dim^2 (A_bar @ x)
    # Matrix-vector multiply: 2 * state_dim * input_dim (B_bar @ u)
    flops_per_step += 2 * state_dim**2 + 2 * state_dim * input_dim
    
    # Output: y = C @ x + D @ u
    # Matrix-vector multiply: 2 * output_dim * state_dim (C @ x)
    # Matrix-vector multiply: 2 * output_dim * input_dim (D @ u)
    flops_per_step += 2 * output_dim * state_dim + 2 * output_dim * input_dim
    
    # Delta computation: delta = softplus(W_delta @ u)
    # Matrix-vector multiply: 2 * state_dim * input_dim
    flops_per_step += 2 * state_dim * input_dim
    
    total_flops = flops_per_step * seq_len
    
    return total_flops


def calculate_flops_attention(
    seq_len: int,
    hidden_dim: int,
) -> int:
    """
    Calculate theoretical FLOPS for attention forward pass.
    
    Args:
        seq_len: Sequence length
        hidden_dim: Hidden dimension
    
    Returns:
        Total FLOPS for forward pass
    """
    # Q, K, V projections: 3 * 2 * seq_len * hidden_dim^2
    flops_qkv = 3 * 2 * seq_len * hidden_dim**2
    
    # Attention scores: Q @ K^T = 2 * seq_len^2 * hidden_dim
    flops_scores = 2 * seq_len**2 * hidden_dim
    
    # Softmax: ~3 * seq_len^2 (exp, sum, divide)
    flops_softmax = 3 * seq_len**2
    
    # Attention output: scores @ V = 2 * seq_len^2 * hidden_dim
    flops_output = 2 * seq_len**2 * hidden_dim
    
    # Output projection: 2 * seq_len * hidden_dim^2
    flops_proj = 2 * seq_len * hidden_dim**2
    
    total_flops = flops_qkv + flops_scores + flops_softmax + flops_output + flops_proj
    
    return total_flops


def calculate_memory_bytes_ssm(
    seq_len: int,
    state_dim: int,
    input_dim: int,
    output_dim: int,
    dtype_bytes: int = 2,  # bf16 or fp16
) -> int:
    """
    Calculate memory bytes transferred for selective SSM forward pass.
    
    Args:
        seq_len: Sequence length
        state_dim: SSM state dimension
        input_dim: Input dimension
        output_dim: Output dimension
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32)
    
    Returns:
        Total bytes transferred
    """
    # Input: seq_len * input_dim
    input_bytes = seq_len * input_dim * dtype_bytes
    
    # Weights: A (state_dim^2), B (state_dim*input_dim), C (output_dim*state_dim), D (output_dim*input_dim)
    # W_delta (state_dim*input_dim)
    weight_bytes = (state_dim**2 + state_dim*input_dim + output_dim*state_dim + 
                    output_dim*input_dim + state_dim*input_dim) * dtype_bytes
    
    # State: state_dim (carried across sequence)
    state_bytes = state_dim * dtype_bytes
    
    # Output: seq_len * output_dim
    output_bytes = seq_len * output_dim * dtype_bytes
    
    # Intermediate activations (approximate)
    # Delta: seq_len * state_dim
    # A_bar, B_bar: seq_len * (state_dim^2 + state_dim*input_dim)
    # x states: seq_len * state_dim
    intermediate_bytes = (seq_len * state_dim + 
                         seq_len * (state_dim**2 + state_dim*input_dim) +
                         seq_len * state_dim) * dtype_bytes
    
    total_bytes = input_bytes + weight_bytes + state_bytes + output_bytes + intermediate_bytes
    
    return total_bytes


def calculate_memory_bytes_attention(
    seq_len: int,
    hidden_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Calculate memory bytes transferred for attention forward pass.
    
    Args:
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32)
    
    Returns:
        Total bytes transferred
    """
    # Input: seq_len * hidden_dim
    input_bytes = seq_len * hidden_dim * dtype_bytes
    
    # Q, K, V: 3 * seq_len * hidden_dim
    qkv_bytes = 3 * seq_len * hidden_dim * dtype_bytes
    
    # Attention scores: seq_len * seq_len
    scores_bytes = seq_len**2 * dtype_bytes
    
    # Attention output: seq_len * hidden_dim
    output_bytes = seq_len * hidden_dim * dtype_bytes
    
    # Weights: Q, K, V, output projections (4 * hidden_dim^2)
    weight_bytes = 4 * hidden_dim**2 * dtype_bytes
    
    # Intermediate activations
    intermediate_bytes = (3 * seq_len * hidden_dim + seq_len**2) * dtype_bytes
    
    total_bytes = input_bytes + qkv_bytes + scores_bytes + output_bytes + weight_bytes + intermediate_bytes
    
    return total_bytes


def calculate_arithmetic_intensity(flops: int, bytes_transferred: int) -> float:
    """
    Calculate arithmetic intensity (FLOPS per byte).
    
    Args:
        flops: Total FLOPS
        bytes_transferred: Total bytes transferred
    
    Returns:
        Arithmetic intensity in FLOPS/byte
    """
    return flops / bytes_transferred


def calculate_achieved_bandwidth(
    bytes_transferred: int,
    time_ms: float,
) -> float:
    """
    Calculate achieved memory bandwidth.
    
    Args:
        bytes_transferred: Total bytes transferred
        time_ms: Time in milliseconds
    
    Returns:
        Achieved bandwidth in GB/s
    """
    bytes_per_second = bytes_transferred / (time_ms / 1000)
    gb_per_second = bytes_per_second / 1e9
    return gb_per_second


def calculate_achieved_tflops(
    flops: int,
    time_ms: float,
) -> float:
    """
    Calculate achieved compute performance.
    
    Args:
        flops: Total FLOPS
        time_ms: Time in milliseconds
    
    Returns:
        Achieved performance in TFLOPS
    """
    flops_per_second = flops / (time_ms / 1000)
    tflops = flops_per_second / 1e12
    return tflops


def calculate_bandwidth_utilization(
    achieved_gb_s: float,
    peak_gb_s: float,
) -> float:
    """
    Calculate memory bandwidth utilization percentage.
    
    Args:
        achieved_gb_s: Achieved bandwidth in GB/s
        peak_gb_s: Theoretical peak bandwidth in GB/s
    
    Returns:
        Utilization percentage (0-100)
    """
    return (achieved_gb_s / peak_gb_s) * 100


def calculate_compute_utilization(
    achieved_tflops: float,
    peak_tflops: float,
) -> float:
    """
    Calculate compute utilization percentage.
    
    Args:
        achieved_tflops: Achieved TFLOPS
        peak_tflops: Theoretical peak TFLOPS
    
    Returns:
        Utilization percentage (0-100)
    """
    return (achieved_tflops / peak_tflops) * 100
