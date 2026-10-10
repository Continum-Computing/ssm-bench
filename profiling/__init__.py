"""
GPU Profiling Infrastructure for SSM Benchmarking

This package provides tools for measuring and analyzing GPU performance
of state-space models (SSMs) vs attention mechanisms.
"""

from profiling.gpu_profiler import (
    get_gpu_specs,
    benchmark_function,
    calculate_flops_ssm,
    calculate_flops_attention,
    calculate_memory_bytes_ssm,
    calculate_memory_bytes_attention,
    calculate_arithmetic_intensity,
    calculate_achieved_bandwidth,
    calculate_achieved_tflops,
    calculate_bandwidth_utilization,
    calculate_compute_utilization,
)

from profiling.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark_sweep,
    save_benchmark_results,
    load_benchmark_results,
)

__all__ = [
    "get_gpu_specs",
    "benchmark_function",
    "calculate_flops_ssm",
    "calculate_flops_attention",
    "calculate_memory_bytes_ssm",
    "calculate_memory_bytes_attention",
    "calculate_arithmetic_intensity",
    "calculate_achieved_bandwidth",
    "calculate_achieved_tflops",
    "calculate_bandwidth_utilization",
    "calculate_compute_utilization",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark_sweep",
    "save_benchmark_results",
    "load_benchmark_results",
]
