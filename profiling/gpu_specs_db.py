"""
Peak hardware specifications for GPUs we benchmark on.

Kept separate from gpu_profiler.py so it can be imported without torch.
That matters for offline work: recomputing derived metrics from stored
results should not need a GPU or a CUDA build of PyTorch.

Values are vendor datasheet figures. Tensor core rates are dense, not
sparse, because sparse rates assume 2:4 structured sparsity that these
workloads do not use. Comparing measured throughput against a sparse
ceiling would understate utilization by roughly 2x.

Bandwidth is the peak for the specific memory configuration, which is why
A100 40GB and 80GB differ. Getting this wrong is not a rounding issue:
the 2026-10-10 runs keyed specs by compute capability, so the Tesla T4
picked up RTX 2080 Ti numbers and every T4 utilization percentage came
out wrong.
"""

from typing import NamedTuple


class PeakSpecs(NamedTuple):
    """Theoretical peak throughput for one GPU model."""

    memory_bandwidth_gb_s: float
    fp32_tflops: float
    fp16_tensor_tflops: float
    bf16_tflops: float
    note: str


# Ordered most specific first. Matching is by substring against the device
# name reported by CUDA, so "A100-SXM4-80GB" has to be tested before "A100".
GPU_PEAK_SPECS = [
    # Turing. Tensor cores do FP16 only, so a bf16 run lands on the FP32
    # CUDA cores and the honest ceiling is the FP32 rate.
    (
        "T4",
        PeakSpecs(320, 8.1, 65.0, 8.1, "Turing, no BF16 tensor core path; bf16 ceiling is FP32"),
    ),
    ("A100-SXM4-80GB", PeakSpecs(2039, 19.5, 312.0, 312.0, "Ampere SXM, 80GB HBM2e")),
    ("A100-SXM4-40GB", PeakSpecs(1555, 19.5, 312.0, 312.0, "Ampere SXM, 40GB HBM2e")),
    ("A100-PCIE-80GB", PeakSpecs(2039, 19.5, 312.0, 312.0, "Ampere PCIe, 80GB")),
    ("A100-PCIE-40GB", PeakSpecs(1555, 19.5, 312.0, 312.0, "Ampere PCIe, 40GB")),
    ("A100 80GB", PeakSpecs(2039, 19.5, 312.0, 312.0, "Ampere, 80GB")),
    ("A100", PeakSpecs(1555, 19.5, 312.0, 312.0, "Ampere, assuming 40GB SXM")),
    ("H100 PCIe", PeakSpecs(2039, 51.0, 756.0, 756.0, "Hopper PCIe")),
    ("H100", PeakSpecs(3350, 67.0, 989.0, 989.0, "Hopper SXM, HBM3")),
    ("L40S", PeakSpecs(864, 91.6, 362.0, 362.0, "Ada Lovelace, GDDR6")),
    ("L4", PeakSpecs(300, 30.3, 121.0, 121.0, "Ada Lovelace, low profile")),
    # Volta. No BF16 tensor core path either.
    ("V100S", PeakSpecs(1134, 16.4, 130.0, 16.4, "Volta, no BF16 tensor core path")),
    ("V100", PeakSpecs(900, 15.7, 125.0, 15.7, "Volta SXM2, no BF16 tensor core path")),
    ("RTX 3090", PeakSpecs(936, 35.6, 142.0, 142.0, "Ampere consumer")),
    ("RTX 2080 TI", PeakSpecs(616, 13.4, 26.9, 13.4, "Turing, no BF16 tensor core path")),
]


def lookup_peak_specs(device_name: str) -> PeakSpecs | None:
    """
    Find datasheet peaks for a CUDA device name.

    Args:
        device_name: Name as reported by torch.cuda.get_device_properties,
            for example "NVIDIA A100-SXM4-40GB" or "Tesla T4".

    Returns:
        Matching PeakSpecs, or None if the GPU is not in the table. Callers
        should treat None as "unknown hardware" rather than guessing, since a
        wrong ceiling silently corrupts every utilization figure downstream.
    """
    normalized = device_name.upper()
    for pattern, specs in GPU_PEAK_SPECS:
        if pattern.upper() in normalized:
            return specs
    return None


def peak_for_dtype(specs: PeakSpecs, dtype: str) -> float:
    """
    Pick the compute ceiling that matches the precision actually run.

    Comparing bf16 throughput against an FP32 ceiling understates
    utilization by an order of magnitude on tensor-core hardware, and the
    reverse overstates it. The benchmark runner already selects peaks this
    way, so derived metrics and plots stay consistent.
    """
    if dtype == "fp32":
        return specs.fp32_tflops
    if dtype == "fp16":
        return specs.fp16_tensor_tflops
    return specs.bf16_tflops


def specs_to_dict(specs: PeakSpecs) -> dict:
    """Serialize peaks for embedding in result metadata."""
    return {
        "memory_bandwidth_gb_s": specs.memory_bandwidth_gb_s,
        "fp32_tflops": specs.fp32_tflops,
        "fp16_tflops": specs.fp16_tensor_tflops,
        "bf16_tflops": specs.bf16_tflops,
        "note": specs.note,
    }
