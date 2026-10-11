# GPU-SSM Mismatch: Why We're Building Custom Silicon

*Oct 10, 2026*

---

## The short version

GPUs are built for matrix multiplication. That's what they're good at. Transformers do a lot of matrix multiplication, so GPUs and transformers work well together.

State-space models (S4, Mamba, Mamba-2, Mamba-3) are different. They process sequences in linear time with a fixed-size recurrent state. The core operation is a small state update, not a big matmul. The GPU runs SSMs correctly, but it's not the right hardware for the job.

We measured this. The gap is real, it's large, and it gets worse on better GPUs. That's why we're building the Continuum SSU-1.

## The problem

Transformers scale as O(N²) in sequence length. Every token attends to every other token. For long contexts, that's expensive.

SSMs fixed this algorithmically. They're O(N) with a fixed-size state. On paper, they should be faster and cheaper.

But there's a second problem the algorithmic fix doesn't address: GPUs are physically designed for dense matrix multiplication. Their architecture (systolic arrays, massive parallel compute, high memory bandwidth for large tensors) is co-designed with the transformer's compute pattern.

SSMs have a different pattern. The core operation is:

```
x[k+1] = Ā · x[k] + B̄ · u[k]      (state update)
y[k]   = C · x[k] + D · u[k]        (output)
```

In selective models (Mamba and beyond), Ā and B̄ depend on the input. The model dynamically chooses what to remember. The computation is a stream of small state updates, not a sequence of large matmuls.

Running an SSM on a GPU is like using a freight train to deliver a package across town. It works, but it's not the right vehicle.

## What we tested

We didn't test whether SSMs are better than transformers. That's already established.

We tested one thing: **do GPUs convert the SSM's arithmetic savings into proportional speed or energy gains?**

We measured three dimensions:

1. **Compute utilization** - how much of the GPU's theoretical FLOP/s actually gets used
2. **Memory bandwidth utilization** - how much of the GPU's bandwidth gets consumed
3. **Energy efficiency** - joules per token vs. the arithmetic cost

We benchmarked on two GPUs: Tesla T4 (Google Colab, free) and A100-SXM4-40GB (Modal, ~$1.80/hour). 36 configurations per GPU: 4 sequence lengths (512, 1024, 2048, 4096) × 3 state dimensions (64, 128, 256) × 3 model types (SSM recurrent, SSM scan, attention).

Baselines: FlashAttention-2 for attention, official Mamba CUDA kernels for SSM. We're comparing against the best available, not deliberately weak code.

## What we found

The gap is large.

**Tesla T4:**
- SSM is 100-200x slower than attention
- SSM uses 0.02-0.15% of bandwidth
- SSM uses < 0.2% of compute
- SSM arithmetic intensity: 3-6 FLOP/byte (memory-bound)
- Attention arithmetic intensity: 146-229 FLOP/byte (compute-bound)

**A100:**
- SSM is 1,000-1,500x slower than attention
- SSM uses 0.01-0.08% of bandwidth (worse than T4)
- SSM uses 0.006% of compute
- Same arithmetic intensity

Here's the critical part: **the gap widens on better hardware.**

A100 has 2.3x more bandwidth than T4, 3x more compute. Attention takes advantage of it (bandwidth utilization jumps from 1.15% to 9.30%). SSM doesn't benefit at all. SSM average latency is identical on both GPUs: 319ms.

If this were a kernel tuning problem, better hardware would close the gap. It doesn't. The mismatch is architectural.

## What would have changed our mind

We set this up as a go/no-go test. If the gap was small (< 40% headroom below roofline), we'd pivot to software optimization or a different workload. If the gap was large (> 60% headroom), we'd proceed with custom silicon.

The gap is 100-1,500x. That's not small.

We also would have reconsidered if:
- SSMs didn't match transformer quality at scale (they do, mostly)
- Hybrid models dominated and attention was >50% of compute (it's not, yet)
- A competitor was already doing this well (they're not)

## Known SSM weaknesses

SSMs aren't universally better than transformers. NVIDIA research shows they lag on:
- In-context learning (learning new patterns from examples in the prompt)
- Copying and lookup (precise token-level recall over long contexts)

These are real tradeoffs, not bugs. They shape what we're building:
- SSU-1 is for workloads where SSMs are strong: long-context understanding, streaming inference, efficient decode
- Hybrid models (SSM + attention, like Jamba) are in scope. The accelerator handles both, but the SSM path is the one we're optimizing.
- We're not trying to make SSMs do things they fundamentally can't do well.

## The path forward

1. ✅ ssm-bench produces GPU gap report (done, Oct 10, 2026)
2. ✅ kalman designs the architecture (done, Oct 10, 2026)
3. ✅ kalman builds the compiler (done, Oct 10, 2026)
4. ✅ kalman writes the RTL (done, Oct 10, 2026)
5. 🔄 kalman prototypes on FPGA (next)
6. ⏳ kalman validates the claim (pending)

Each step is gated by the previous step. We don't move forward until the tests pass.

## Reproducing this

All the data is in this repo.

```bash
git clone https://github.com/Continuum-Computiing/ssm-bench.git
cd ssm-bench
make setup
make test          # correctness tests
make profile       # GPU profiling (requires NVIDIA GPU)
```

Raw results:
- [Tesla T4](profiling/results/t4-colab-2026-10-10/)
- [A100](profiling/results/a100-modal-2026-10-10/)
- [T4 vs A100 comparison](profiling/results/analysis/gpu_comparison_t4_vs_a100.md)

## What this isn't

This isn't a business plan. It doesn't address market size, funding, or go-to-market.

This isn't an architecture spec. It doesn't describe the accelerator design in detail.

This isn't a benchmark methodology paper. See the main README for methodology details.

This is a statement of what we believed, what we tested, what we found, and what we're doing about it.
