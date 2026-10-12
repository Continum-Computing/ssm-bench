"""
Visualization utilities for benchmark results.

This module generates charts and reports from benchmark results,
helping to visualize performance characteristics and identify
the GPU-SSM mismatch.
"""

import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_latency_vs_sequence_length(results: Dict, output_path: str = "latency_vs_seq_len.png"):
    """
    Plot latency vs sequence length for all model types.
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save the plot
    """
    data = results['results']
    
    # Group by model type and sequence length
    ssm_recurrent = {}
    ssm_scan = {}
    attention = {}
    
    for r in data:
        seq_len = r['config']['seq_len']
        latency = r['latency_stats']['mean_ms']
        
        if r['model_type'] == 'ssm_recurrent':
            if seq_len not in ssm_recurrent:
                ssm_recurrent[seq_len] = []
            ssm_recurrent[seq_len].append(latency)
        elif r['model_type'] == 'ssm_scan':
            if seq_len not in ssm_scan:
                ssm_scan[seq_len] = []
            ssm_scan[seq_len].append(latency)
        elif r['model_type'] == 'attention':
            if seq_len not in attention:
                attention[seq_len] = []
            attention[seq_len].append(latency)
    
    # Calculate averages
    seq_lens = sorted(ssm_recurrent.keys())
    ssm_rec_avg = [np.mean(ssm_recurrent[s]) for s in seq_lens]
    ssm_scan_avg = [np.mean(ssm_scan[s]) for s in seq_lens]
    attn_avg = [np.mean(attention[s]) for s in seq_lens]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, ssm_rec_avg, 'o-', label='SSM Recurrent', linewidth=2, markersize=8)
    plt.plot(seq_lens, ssm_scan_avg, 's-', label='SSM Scan', linewidth=2, markersize=8)
    plt.plot(seq_lens, attn_avg, '^-', label='Attention', linewidth=2, markersize=8)
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Latency vs Sequence Length', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved latency plot to {output_path}")


def plot_bandwidth_utilization(results: Dict, output_path: str = "bandwidth_utilization.png"):
    """
    Plot bandwidth utilization vs sequence length.
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save the plot
    """
    data = results['results']
    
    # Group by model type and sequence length
    ssm_recurrent = {}
    ssm_scan = {}
    attention = {}
    
    for r in data:
        seq_len = r['config']['seq_len']
        bw_util = r['bandwidth_utilization_pct']
        
        if r['model_type'] == 'ssm_recurrent':
            if seq_len not in ssm_recurrent:
                ssm_recurrent[seq_len] = []
            ssm_recurrent[seq_len].append(bw_util)
        elif r['model_type'] == 'ssm_scan':
            if seq_len not in ssm_scan:
                ssm_scan[seq_len] = []
            ssm_scan[seq_len].append(bw_util)
        elif r['model_type'] == 'attention':
            if seq_len not in attention:
                attention[seq_len] = []
            attention[seq_len].append(bw_util)
    
    # Calculate averages
    seq_lens = sorted(ssm_recurrent.keys())
    ssm_rec_avg = [np.mean(ssm_recurrent[s]) for s in seq_lens]
    ssm_scan_avg = [np.mean(ssm_scan[s]) for s in seq_lens]
    attn_avg = [np.mean(attention[s]) for s in seq_lens]

    # Calculate dynamic y-axis max based on data
    all_values = ssm_rec_avg + ssm_scan_avg + attn_avg
    y_max = max(all_values) * 1.2  # 20% padding
    y_max = max(y_max, 1.0)  # At least 1%

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, ssm_rec_avg, 'o-', label='SSM Recurrent', linewidth=2, markersize=8)
    plt.plot(seq_lens, ssm_scan_avg, 's-', label='SSM Scan', linewidth=2, markersize=8)
    plt.plot(seq_lens, attn_avg, '^-', label='Attention', linewidth=2, markersize=8)

    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Bandwidth Utilization (%)', fontsize=12)
    plt.title('Memory Bandwidth Utilization vs Sequence Length', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved bandwidth utilization plot to {output_path}")


def plot_compute_utilization(results: Dict, output_path: str = "compute_utilization.png"):
    """
    Plot compute utilization vs sequence length.
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save the plot
    """
    data = results['results']
    
    # Group by model type and sequence length
    ssm_recurrent = {}
    ssm_scan = {}
    attention = {}
    
    for r in data:
        seq_len = r['config']['seq_len']
        compute_util = r['compute_utilization_pct']
        
        if r['model_type'] == 'ssm_recurrent':
            if seq_len not in ssm_recurrent:
                ssm_recurrent[seq_len] = []
            ssm_recurrent[seq_len].append(compute_util)
        elif r['model_type'] == 'ssm_scan':
            if seq_len not in ssm_scan:
                ssm_scan[seq_len] = []
            ssm_scan[seq_len].append(compute_util)
        elif r['model_type'] == 'attention':
            if seq_len not in attention:
                attention[seq_len] = []
            attention[seq_len].append(compute_util)
    
    # Calculate averages
    seq_lens = sorted(ssm_recurrent.keys())
    ssm_rec_avg = [np.mean(ssm_recurrent[s]) for s in seq_lens]
    ssm_scan_avg = [np.mean(ssm_scan[s]) for s in seq_lens]
    attn_avg = [np.mean(attention[s]) for s in seq_lens]

    # Calculate dynamic y-axis max based on data
    all_values = ssm_rec_avg + ssm_scan_avg + attn_avg
    y_max = max(all_values) * 1.2  # 20% padding
    y_max = max(y_max, 1.0)  # At least 1%

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, ssm_rec_avg, 'o-', label='SSM Recurrent', linewidth=2, markersize=8)
    plt.plot(seq_lens, ssm_scan_avg, 's-', label='SSM Scan', linewidth=2, markersize=8)
    plt.plot(seq_lens, attn_avg, '^-', label='Attention', linewidth=2, markersize=8)

    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Compute Utilization (%)', fontsize=12)
    plt.title('Compute Utilization vs Sequence Length', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved compute utilization plot to {output_path}")


def plot_roofline_model(results: Dict, output_path: str = "roofline_model.png"):
    """
    Plot roofline model showing arithmetic intensity vs achieved performance.
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save the plot
    """
    data = results['results']
    gpu_info = results['metadata']['gpu']
    
    # Extract data points
    ssm_rec_points = []
    ssm_scan_points = []
    attn_points = []
    
    for r in data:
        ai = r['arithmetic_intensity']
        perf = r['achieved_tflops']
        
        if r['model_type'] == 'ssm_recurrent':
            ssm_rec_points.append((ai, perf))
        elif r['model_type'] == 'ssm_scan':
            ssm_scan_points.append((ai, perf))
        elif r['model_type'] == 'attention':
            attn_points.append((ai, perf))
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot roofline
    peak_bw = gpu_info['memory_bandwidth_gb_s']
    peak_compute = gpu_info['fp32_tflops']
    
    # Memory bandwidth roofline
    ai_range = np.logspace(-1, 3, 100)
    mem_roofline = peak_bw * ai_range / 1000  # Convert to TFLOPS
    
    plt.plot(ai_range, mem_roofline, 'k--', label='Memory Bandwidth Limit', linewidth=2)
    plt.axhline(y=peak_compute, color='r', linestyle='--', label='Compute Limit', linewidth=2)
    
    # Plot data points
    if ssm_rec_points:
        ai_rec, perf_rec = zip(*ssm_rec_points)
        plt.scatter(ai_rec, perf_rec, c='blue', marker='o', s=100, label='SSM Recurrent', zorder=5)
    
    if ssm_scan_points:
        ai_scan, perf_scan = zip(*ssm_scan_points)
        plt.scatter(ai_scan, perf_scan, c='green', marker='s', s=100, label='SSM Scan', zorder=5)
    
    if attn_points:
        ai_attn, perf_attn = zip(*attn_points)
        plt.scatter(ai_attn, perf_attn, c='red', marker='^', s=100, label='Attention', zorder=5)
    
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPS)', fontsize=12)
    plt.title('Roofline Model: Performance vs Arithmetic Intensity', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved roofline model to {output_path}")


def generate_summary_report(results: Dict, output_path: str = "benchmark_summary.txt"):
    """
    Generate a text summary report of benchmark results.
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save the report
    """
    data = results['results']
    gpu_info = results['metadata']['gpu']
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SSM vs Attention Benchmark Summary Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("GPU Information:\n")
        f.write(f"  Name: {gpu_info['name']}\n")
        f.write(f"  Compute Capability: {gpu_info['compute_capability']}\n")
        f.write(f"  Total Memory: {gpu_info['total_memory_gb']:.1f} GB\n")
        f.write(f"  Peak Memory Bandwidth: {gpu_info['memory_bandwidth_gb_s']:.0f} GB/s\n")
        f.write(f"  Peak FP32 Performance: {gpu_info['fp32_tflops']:.1f} TFLOPS\n")
        f.write(f"  Peak BF16 Performance: {gpu_info['bf16_tflops']:.1f} TFLOPS\n")
        f.write("\n")
        
        f.write("Benchmark Metadata:\n")
        f.write(f"  Timestamp: {results['metadata']['timestamp']}\n")
        f.write(f"  Python Version: {results['metadata']['python_version']}\n")
        f.write(f"  PyTorch Version: {results['metadata']['torch_version']}\n")
        f.write(f"  CUDA Version: {results['metadata']['cuda_version']}\n")
        f.write("\n")
        
        # Group results by model type
        ssm_recurrent = [r for r in data if r['model_type'] == 'ssm_recurrent']
        ssm_scan = [r for r in data if r['model_type'] == 'ssm_scan']
        attention = [r for r in data if r['model_type'] == 'attention']
        
        f.write("=" * 80 + "\n")
        f.write("Performance Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Latency comparison
        f.write("Average Latency (ms):\n")
        avg_ssm_rec = np.mean([r['latency_stats']['mean_ms'] for r in ssm_recurrent])
        avg_ssm_scan = np.mean([r['latency_stats']['mean_ms'] for r in ssm_scan])
        avg_attn = np.mean([r['latency_stats']['mean_ms'] for r in attention])
        
        f.write(f"  SSM Recurrent: {avg_ssm_rec:.3f} ms\n")
        f.write(f"  SSM Scan:      {avg_ssm_scan:.3f} ms\n")
        f.write(f"  Attention:     {avg_attn:.3f} ms\n")
        f.write("\n")
        
        # Bandwidth utilization
        f.write("Average Bandwidth Utilization (%):\n")
        avg_ssm_rec_bw = np.mean([r['bandwidth_utilization_pct'] for r in ssm_recurrent])
        avg_ssm_scan_bw = np.mean([r['bandwidth_utilization_pct'] for r in ssm_scan])
        avg_attn_bw = np.mean([r['bandwidth_utilization_pct'] for r in attention])
        
        f.write(f"  SSM Recurrent: {avg_ssm_rec_bw:.1f}%\n")
        f.write(f"  SSM Scan:      {avg_ssm_scan_bw:.1f}%\n")
        f.write(f"  Attention:     {avg_attn_bw:.1f}%\n")
        f.write("\n")
        
        # Compute utilization
        f.write("Average Compute Utilization (%):\n")
        avg_ssm_rec_comp = np.mean([r['compute_utilization_pct'] for r in ssm_recurrent])
        avg_ssm_scan_comp = np.mean([r['compute_utilization_pct'] for r in ssm_scan])
        avg_attn_comp = np.mean([r['compute_utilization_pct'] for r in attention])
        
        f.write(f"  SSM Recurrent: {avg_ssm_rec_comp:.1f}%\n")
        f.write(f"  SSM Scan:      {avg_ssm_scan_comp:.1f}%\n")
        f.write(f"  Attention:     {avg_attn_comp:.1f}%\n")
        f.write("\n")
        
        # Key findings
        f.write("=" * 80 + "\n")
        f.write("Key Findings\n")
        f.write("=" * 80 + "\n\n")
        
        # Calculate speedup
        speedup_rec = avg_attn / avg_ssm_rec
        speedup_scan = avg_attn / avg_ssm_scan
        
        f.write(f"SSM Speedup vs Attention:\n")
        f.write(f"  Recurrent: {speedup_rec:.2f}x faster\n")
        f.write(f"  Scan:      {speedup_scan:.2f}x faster\n")
        f.write("\n")
        
        # GPU utilization gap
        f.write("GPU Utilization Gap:\n")
        f.write(f"  SSM Recurrent bandwidth utilization: {avg_ssm_rec_bw:.1f}%\n")
        f.write(f"  Attention bandwidth utilization:     {avg_attn_bw:.1f}%\n")
        f.write(f"  Gap: {avg_attn_bw - avg_ssm_rec_bw:.1f} percentage points\n")
        f.write("\n")
        
        f.write("Interpretation:\n")
        if avg_ssm_rec_bw < avg_attn_bw * 0.7:
            f.write("  ✓ SSM shows significantly lower bandwidth utilization than attention,\n")
            f.write("    suggesting GPUs are not optimized for SSM workloads.\n")
            f.write("    This supports the hypothesis for custom SSM accelerator design.\n")
        else:
            f.write("  ✗ SSM bandwidth utilization is comparable to attention,\n")
            f.write("    suggesting GPUs handle SSM workloads reasonably well.\n")
            f.write("    Custom accelerator may not provide significant benefits.\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved summary report to {output_path}")


def generate_all_visualizations(results_path: str, output_dir: str = "profiling/results"):
    """
    Generate all visualizations and reports from benchmark results.
    
    Args:
        results_path: Path to benchmark results JSON
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = load_results(results_path)
    
    print("Generating visualizations...")
    print()
    
    plot_latency_vs_sequence_length(results, f"{output_dir}/latency_vs_seq_len.png")
    plot_bandwidth_utilization(results, f"{output_dir}/bandwidth_utilization.png")
    plot_compute_utilization(results, f"{output_dir}/compute_utilization.png")
    plot_roofline_model(results, f"{output_dir}/roofline_model.png")
    generate_summary_report(results, f"{output_dir}/benchmark_summary.txt")
    
    print()
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
