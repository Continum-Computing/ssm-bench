"""
Benchmark Runner - Orchestrate SSM and attention benchmarks.

This module runs systematic benchmarks across different configurations
(sequence length, batch size, model dimensions) and collects performance
metrics for both SSM and attention mechanisms.
"""

import json
import os
import platform

# Import from model package
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pytorch_impl.selective_ssm import (
    compute_delta,
    selective_ssm_recurrent,
    selective_ssm_scan,
)
from profiling.gpu_profiler import (
    benchmark_function,
    calculate_achieved_bandwidth,
    calculate_achieved_tflops,
    calculate_arithmetic_intensity,
    calculate_bandwidth_utilization,
    calculate_compute_utilization,
    calculate_flops_attention,
    calculate_flops_ssm,
    calculate_memory_bytes_attention,
    calculate_memory_bytes_ssm,
    get_gpu_specs,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    seq_len: int
    batch_size: int
    state_dim: int
    input_dim: int
    output_dim: int
    dtype: str  # "fp32", "fp16", "bf16"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    model_type: str  # "ssm_recurrent", "ssm_scan", "attention"
    latency_stats: dict[str, float]  # mean, std, min, max, median, p95, p99
    flops: int
    bytes_transferred: int
    arithmetic_intensity: float  # FLOPS/byte
    achieved_bandwidth_gb_s: float
    achieved_tflops: float
    bandwidth_utilization_pct: float
    compute_utilization_pct: float
    
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "model_type": self.model_type,
            "latency_stats": self.latency_stats,
            "flops": self.flops,
            "bytes_transferred": self.bytes_transferred,
            "arithmetic_intensity": self.arithmetic_intensity,
            "achieved_bandwidth_gb_s": self.achieved_bandwidth_gb_s,
            "achieved_tflops": self.achieved_tflops,
            "bandwidth_utilization_pct": self.bandwidth_utilization_pct,
            "compute_utilization_pct": self.compute_utilization_pct,
        }


def create_ssm_inputs(config: BenchmarkConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Create input tensors for SSM benchmark."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[config.dtype]
    
    # Create random inputs
    u = torch.randn(config.seq_len, config.input_dim, dtype=dtype, device=device)
    A = torch.randn(config.state_dim, config.state_dim, dtype=dtype, device=device) * 0.1
    B = torch.randn(config.state_dim, config.input_dim, dtype=dtype, device=device)
    C = torch.randn(config.output_dim, config.state_dim, dtype=dtype, device=device)
    D = torch.randn(config.output_dim, config.input_dim, dtype=dtype, device=device)
    W_delta = torch.randn(config.state_dim, config.input_dim, dtype=dtype, device=device)
    b_delta = torch.randn(config.state_dim, dtype=dtype, device=device)
    
    return {
        "u": u,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "W_delta": W_delta,
        "b_delta": b_delta,
    }


def create_attention_inputs(config: BenchmarkConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Create input tensors for attention benchmark."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[config.dtype]
    
    # For attention, we use hidden_dim = input_dim = output_dim
    hidden_dim = config.input_dim
    
    # Create random input
    x = torch.randn(config.seq_len, hidden_dim, dtype=dtype, device=device)
    
    # Q, K, V projection weights
    W_q = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
    W_k = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
    W_v = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
    W_o = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
    
    return {
        "x": x,
        "W_q": W_q,
        "W_k": W_k,
        "W_v": W_v,
        "W_o": W_o,
    }


def simple_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, 
                     W_v: torch.Tensor, W_o: torch.Tensor) -> torch.Tensor:
    """
    Simple attention implementation for benchmarking.
    
    Args:
        x: Input tensor [seq_len, hidden_dim]
        W_q, W_k, W_v, W_o: Projection weights
    
    Returns:
        Output tensor [seq_len, hidden_dim]
    """
    # Project to Q, K, V
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    
    # Compute attention scores
    scores = Q @ K.T / (K.shape[-1] ** 0.5)
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention
    output = attn_weights @ V
    
    # Output projection
    output = output @ W_o
    
    return output


def benchmark_ssm_recurrent(config: BenchmarkConfig, device: torch.device) -> BenchmarkResult:
    """Benchmark SSM recurrent implementation."""
    inputs = create_ssm_inputs(config, device)
    
    # Define the function to benchmark
    def run_ssm_recurrent():
        delta = compute_delta(inputs["u"], inputs["W_delta"], inputs["b_delta"])
        y, x_final = selective_ssm_recurrent(
            inputs["u"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["D"],
            delta,
        )
        return y
    
    # Run benchmark
    latency_stats = benchmark_function(run_ssm_recurrent, warmup_runs=10, benchmark_runs=100)
    
    # Calculate metrics
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[config.dtype]
    flops = calculate_flops_ssm(config.seq_len, config.state_dim, config.input_dim, config.output_dim)
    bytes_transferred = calculate_memory_bytes_ssm(
        config.seq_len, config.state_dim, config.input_dim, config.output_dim, dtype_bytes
    )
    
    arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_transferred)
    achieved_bandwidth = calculate_achieved_bandwidth(bytes_transferred, latency_stats["mean_ms"])
    achieved_tflops = calculate_achieved_tflops(flops, latency_stats["mean_ms"])
    
    # Get GPU specs for utilization calculation
    gpu_specs = get_gpu_specs()
    bandwidth_utilization = calculate_bandwidth_utilization(
        achieved_bandwidth, gpu_specs.memory_bandwidth_gb_s
    )
    compute_utilization = calculate_compute_utilization(
        achieved_tflops, gpu_specs.fp32_tflops if config.dtype == "fp32" else gpu_specs.bf16_tflops
    )
    
    return BenchmarkResult(
        config=config,
        model_type="ssm_recurrent",
        latency_stats=latency_stats,
        flops=flops,
        bytes_transferred=bytes_transferred,
        arithmetic_intensity=arithmetic_intensity,
        achieved_bandwidth_gb_s=achieved_bandwidth,
        achieved_tflops=achieved_tflops,
        bandwidth_utilization_pct=bandwidth_utilization,
        compute_utilization_pct=compute_utilization,
    )


def benchmark_ssm_scan(config: BenchmarkConfig, device: torch.device) -> BenchmarkResult:
    """Benchmark SSM scan implementation."""
    inputs = create_ssm_inputs(config, device)
    
    # Define the function to benchmark
    def run_ssm_scan():
        delta = compute_delta(inputs["u"], inputs["W_delta"], inputs["b_delta"])
        y, x_final = selective_ssm_scan(
            inputs["u"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            inputs["D"],
            delta,
        )
        return y
    
    # Run benchmark
    latency_stats = benchmark_function(run_ssm_scan, warmup_runs=10, benchmark_runs=100)
    
    # Calculate metrics
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[config.dtype]
    flops = calculate_flops_ssm(config.seq_len, config.state_dim, config.input_dim, config.output_dim)
    bytes_transferred = calculate_memory_bytes_ssm(
        config.seq_len, config.state_dim, config.input_dim, config.output_dim, dtype_bytes
    )
    
    arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_transferred)
    achieved_bandwidth = calculate_achieved_bandwidth(bytes_transferred, latency_stats["mean_ms"])
    achieved_tflops = calculate_achieved_tflops(flops, latency_stats["mean_ms"])
    
    # Get GPU specs for utilization calculation
    gpu_specs = get_gpu_specs()
    bandwidth_utilization = calculate_bandwidth_utilization(
        achieved_bandwidth, gpu_specs.memory_bandwidth_gb_s
    )
    compute_utilization = calculate_compute_utilization(
        achieved_tflops, gpu_specs.fp32_tflops if config.dtype == "fp32" else gpu_specs.bf16_tflops
    )
    
    return BenchmarkResult(
        config=config,
        model_type="ssm_scan",
        latency_stats=latency_stats,
        flops=flops,
        bytes_transferred=bytes_transferred,
        arithmetic_intensity=arithmetic_intensity,
        achieved_bandwidth_gb_s=achieved_bandwidth,
        achieved_tflops=achieved_tflops,
        bandwidth_utilization_pct=bandwidth_utilization,
        compute_utilization_pct=compute_utilization,
    )


def benchmark_attention(config: BenchmarkConfig, device: torch.device) -> BenchmarkResult:
    """Benchmark attention implementation."""
    inputs = create_attention_inputs(config, device)
    
    # Define the function to benchmark
    def run_attention():
        return simple_attention(
            inputs["x"],
            inputs["W_q"],
            inputs["W_k"],
            inputs["W_v"],
            inputs["W_o"],
        )
    
    # Run benchmark
    latency_stats = benchmark_function(run_attention, warmup_runs=10, benchmark_runs=100)
    
    # Calculate metrics
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[config.dtype]
    hidden_dim = config.input_dim
    flops = calculate_flops_attention(config.seq_len, hidden_dim)
    bytes_transferred = calculate_memory_bytes_attention(config.seq_len, hidden_dim, dtype_bytes)
    
    arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_transferred)
    achieved_bandwidth = calculate_achieved_bandwidth(bytes_transferred, latency_stats["mean_ms"])
    achieved_tflops = calculate_achieved_tflops(flops, latency_stats["mean_ms"])
    
    # Get GPU specs for utilization calculation
    gpu_specs = get_gpu_specs()
    bandwidth_utilization = calculate_bandwidth_utilization(
        achieved_bandwidth, gpu_specs.memory_bandwidth_gb_s
    )
    compute_utilization = calculate_compute_utilization(
        achieved_tflops, gpu_specs.fp32_tflops if config.dtype == "fp32" else gpu_specs.bf16_tflops
    )
    
    return BenchmarkResult(
        config=config,
        model_type="attention",
        latency_stats=latency_stats,
        flops=flops,
        bytes_transferred=bytes_transferred,
        arithmetic_intensity=arithmetic_intensity,
        achieved_bandwidth_gb_s=achieved_bandwidth,
        achieved_tflops=achieved_tflops,
        bandwidth_utilization_pct=bandwidth_utilization,
        compute_utilization_pct=compute_utilization,
    )


def run_benchmark_sweep(
    seq_lens: list[int] = [512, 1024, 2048, 4096],
    state_dims: list[int] = [64, 128, 256],
    input_dim: int = 256,
    output_dim: int = 256,
    dtype: str = "bf16",
    device_id: int = 0,
) -> list[BenchmarkResult]:
    """
    Run a sweep of benchmarks across different configurations.
    
    Args:
        seq_lens: List of sequence lengths to test
        state_dims: List of state dimensions to test (for SSM)
        input_dim: Input dimension (fixed)
        output_dim: Output dimension (fixed)
        dtype: Data type ("fp32", "fp16", "bf16")
        device_id: CUDA device ID
    
    Returns:
        List of BenchmarkResult objects
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - GPU required for benchmarking")
    
    device = torch.device(f"cuda:{device_id}")
    results = []
    
    # Get GPU specs once
    gpu_specs = get_gpu_specs(device_id)
    print(f"GPU: {gpu_specs.name}")
    print(f"Compute Capability: {gpu_specs.compute_capability[0]}.{gpu_specs.compute_capability[1]}")
    print(f"Memory: {gpu_specs.total_memory_gb:.1f} GB")
    print(f"Peak Bandwidth: {gpu_specs.memory_bandwidth_gb_s:.0f} GB/s")
    print(f"Peak FP32: {gpu_specs.fp32_tflops:.1f} TFLOPS")
    print(f"Peak BF16: {gpu_specs.bf16_tflops:.1f} TFLOPS")
    print()
    
    total_configs = len(seq_lens) * len(state_dims)
    current = 0
    
    for seq_len in seq_lens:
        for state_dim in state_dims:
            current += 1
            print(f"[{current}/{total_configs}] seq_len={seq_len}, state_dim={state_dim}")
            
            config = BenchmarkConfig(
                seq_len=seq_len,
                batch_size=1,  # Fixed for now
                state_dim=state_dim,
                input_dim=input_dim,
                output_dim=output_dim,
                dtype=dtype,
            )
            
            # Benchmark SSM recurrent
            print("  - SSM recurrent...")
            result_ssm_rec = benchmark_ssm_recurrent(config, device)
            results.append(result_ssm_rec)
            print(f"    Latency: {result_ssm_rec.latency_stats['mean_ms']:.3f} ms")
            print(f"    Bandwidth: {result_ssm_rec.achieved_bandwidth_gb_s:.1f} GB/s ({result_ssm_rec.bandwidth_utilization_pct:.1f}%)")
            
            # Benchmark SSM scan
            print("  - SSM scan...")
            result_ssm_scan = benchmark_ssm_scan(config, device)
            results.append(result_ssm_scan)
            print(f"    Latency: {result_ssm_scan.latency_stats['mean_ms']:.3f} ms")
            print(f"    Bandwidth: {result_ssm_scan.achieved_bandwidth_gb_s:.1f} GB/s ({result_ssm_scan.bandwidth_utilization_pct:.1f}%)")
            
            # Benchmark attention (use input_dim as hidden_dim)
            print("  - Attention...")
            result_attn = benchmark_attention(config, device)
            results.append(result_attn)
            print(f"    Latency: {result_attn.latency_stats['mean_ms']:.3f} ms")
            print(f"    Bandwidth: {result_attn.achieved_bandwidth_gb_s:.1f} GB/s ({result_attn.bandwidth_utilization_pct:.1f}%)")
            
            print()
    
    return results


def save_benchmark_results(results: list[BenchmarkResult], output_path: str):
    """
    Save benchmark results to JSON file.
    
    Args:
        results: List of BenchmarkResult objects
        output_path: Path to output JSON file
    """
    # Get GPU specs
    gpu_specs = get_gpu_specs()
    
    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_specs.to_dict() if gpu_specs else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    # Build results
    results_dict = {
        "metadata": metadata,
        "results": [r.to_dict() for r in results],
    }
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {output_path}")


def load_benchmark_results(input_path: str) -> dict:
    """
    Load benchmark results from JSON file.
    
    Args:
        input_path: Path to input JSON file
    
    Returns:
        Dictionary with metadata and results
    """
    with open(input_path) as f:
        return json.load(f)
