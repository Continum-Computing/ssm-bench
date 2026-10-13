#!/usr/bin/env python3
"""
Main benchmark script for SSM vs Attention performance comparison.

This script runs systematic benchmarks across different configurations
and generates detailed performance reports.

Usage:
    python profiling/run_benchmarks.py [--output results.json] [--device 0]
"""

import argparse
import os
import sys

# Add project root to path so we can import from model and profiling
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from profiling.benchmark_runner import run_benchmark_sweep, save_benchmark_results
from profiling.gpu_profiler import get_gpu_specs


def main():
    parser = argparse.ArgumentParser(description="Run SSM vs Attention benchmarks")
    parser.add_argument(
        "--output",
        type=str,
        default="profiling/results/benchmark_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--state-dims",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="SSM state dimensions to test",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=256,
        help="Input dimension",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
        help="Output dimension",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Data type for computation",
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU is required for benchmarking.")
        print("Please ensure you have a compatible NVIDIA GPU and CUDA installed.")
        sys.exit(1)
    
    # Print GPU info
    gpu_specs = get_gpu_specs(args.device)
    if gpu_specs is None:
        print(f"ERROR: Could not query GPU specs for device {args.device}")
        sys.exit(1)
    
    print("=" * 70)
    print("SSM vs Attention Benchmark Suite")
    print("=" * 70)
    print()
    print(f"GPU: {gpu_specs.name}")
    print(f"Compute Capability: {gpu_specs.compute_capability[0]}.{gpu_specs.compute_capability[1]}")
    print(f"Total Memory: {gpu_specs.total_memory_gb:.1f} GB")
    print(f"Peak Memory Bandwidth: {gpu_specs.memory_bandwidth_gb_s:.0f} GB/s")
    print(f"Peak FP32 Performance: {gpu_specs.fp32_tflops:.1f} TFLOPS")
    print(f"Peak BF16 Performance: {gpu_specs.bf16_tflops:.1f} TFLOPS")
    print()
    print("Benchmark Configuration:")
    print(f"  Sequence lengths: {args.seq_lens}")
    print(f"  State dimensions: {args.state_dims}")
    print(f"  Input dimension: {args.input_dim}")
    print(f"  Output dimension: {args.output_dim}")
    print(f"  Data type: {args.dtype}")
    print()
    print("=" * 70)
    print()
    
    # Run benchmarks
    print("Starting benchmarks...")
    print()
    
    results = run_benchmark_sweep(
        seq_lens=args.seq_lens,
        state_dims=args.state_dims,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        dtype=args.dtype,
        device_id=args.device,
    )
    
    # Save results
    save_benchmark_results(results, args.output)
    
    print()
    print("=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print()
    
    # Print summary statistics
    ssm_recurrent_results = [r for r in results if r.model_type == "ssm_recurrent"]
    ssm_scan_results = [r for r in results if r.model_type == "ssm_scan"]
    attention_results = [r for r in results if r.model_type == "attention"]
    
    print(f"Total configurations tested: {len(results) // 3}")
    print(f"  - SSM recurrent: {len(ssm_recurrent_results)} runs")
    print(f"  - SSM scan: {len(ssm_scan_results)} runs")
    print(f"  - Attention: {len(attention_results)} runs")
    print()
    
    # Calculate averages
    avg_ssm_rec_latency = sum(r.latency_stats["mean_ms"] for r in ssm_recurrent_results) / len(ssm_recurrent_results)
    avg_ssm_scan_latency = sum(r.latency_stats["mean_ms"] for r in ssm_scan_results) / len(ssm_scan_results)
    avg_attn_latency = sum(r.latency_stats["mean_ms"] for r in attention_results) / len(attention_results)
    
    print("Average Latency:")
    print(f"  - SSM recurrent: {avg_ssm_rec_latency:.3f} ms")
    print(f"  - SSM scan: {avg_ssm_scan_latency:.3f} ms")
    print(f"  - Attention: {avg_attn_latency:.3f} ms")
    print()
    
    avg_ssm_rec_bw = sum(r.bandwidth_utilization_pct for r in ssm_recurrent_results) / len(ssm_recurrent_results)
    avg_ssm_scan_bw = sum(r.bandwidth_utilization_pct for r in ssm_scan_results) / len(ssm_scan_results)
    avg_attn_bw = sum(r.bandwidth_utilization_pct for r in attention_results) / len(attention_results)
    
    print("Average Bandwidth Utilization:")
    print(f"  - SSM recurrent: {avg_ssm_rec_bw:.1f}%")
    print(f"  - SSM scan: {avg_ssm_scan_bw:.1f}%")
    print(f"  - Attention: {avg_attn_bw:.1f}%")
    print()
    
    avg_ssm_rec_compute = sum(r.compute_utilization_pct for r in ssm_recurrent_results) / len(ssm_recurrent_results)
    avg_ssm_scan_compute = sum(r.compute_utilization_pct for r in ssm_scan_results) / len(ssm_scan_results)
    avg_attn_compute = sum(r.compute_utilization_pct for r in attention_results) / len(attention_results)
    
    print("Average Compute Utilization:")
    print(f"  - SSM recurrent: {avg_ssm_rec_compute:.1f}%")
    print(f"  - SSM scan: {avg_ssm_scan_compute:.1f}%")
    print(f"  - Attention: {avg_attn_compute:.1f}%")
    print()
    
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
