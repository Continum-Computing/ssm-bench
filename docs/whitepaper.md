# GPU-SSM Mismatch: Why We're Building Custom Silicon

*Oct 10, 2026. Corrected Oct 13, 2026 after an honest review of what the benchmarks actually measure.*

---

## The short version

GPUs are built for matrix multiplication. That's what they're good at. Transformers do a lot of matrix multiplication, so GPUs and transformers work well together.

State-space models (S4, Mamba, Mamba-2, Mamba-3) are different. They process sequences in linear time with a fixed-size recurrent state. The core operation is a small state update, not a big matmul.

Our hypothesis: the GPU runs SSMs correctly, but it can't turn the algorithm's arithmetic savings into proportional speed or energy savings. If that's true, a custom processor built around the recurrence pattern is worth building. That's the Continuum SSU-1.

We started measuring. What we have so far is a set of reference implementation benchmarks on two GPUs. They are not the test of the hypothesis yet, and this document says so everywhere it matters. The real test needs fused kernels on both sides, and those are in progress.

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

Running an SSM on a GPU is like using a freight train to deliver a package across town. It works, but it's not the right vehicle. That's the intuition. Intuition is not evidence, which is why we built ssm-bench.

## What we're testing

We aren't testing whether SSMs are better than transformers. That's largely settled elsewhere.

We're testing one thing: **do GPUs convert the SSM's arithmetic savings into proportional speed or energy gains?**

Three dimensions matter:

1. **Compute utilization** - how much of the GPU's theoretical FLOP/s actually gets used
2. **Memory bandwidth utilization** - how much of the GPU's bandwidth gets consumed
3. **Energy efficiency** - joules per token against the arithmetic cost

Dimension 3 is specified but not implemented. There is no power telemetry code in this repo yet, so we have no energy numbers and we make no throughput-per-watt claim.

## What we ran

Two GPUs: Tesla T4 (Google Colab, free) and A100-SXM4-40GB (Modal, about $1.80/hour). 36 configurations per GPU: 4 sequence lengths (512, 1024, 2048, 4096) × 3 state dimensions (64, 128, 256) × 3 model types (SSM recurrent, SSM scan, attention).

Both sides are our own reference implementations, written to be correct and readable:

- **SSM:** a Python for-loop over time steps. Each step is a handful of tiny GPU operations, so a 4096-step sequence means thousands of kernel launches.
- **Attention:** plain matmul, softmax, matmul. No FlashAttention.

Batch size 1, bf16, no CUDA graphs.

**This is the part to hold onto.** We are comparing naive SSM code against naive attention code. That comparison mostly measures Python loop and launch overhead. It does not measure how a GPU handles a well-implemented SSM.

## What we found

**Tesla T4:**
- SSM is 100-200x slower than attention
- SSM uses 0.02-0.15% of bandwidth
- SSM uses < 0.2% of compute
- SSM arithmetic intensity: 3-6 FLOP/byte (memory-bound)
- Attention arithmetic intensity: 146-229 FLOP/byte (compute-bound)

**A100:**
- SSM is 1,000-1,500x slower than attention
- SSM uses 0.01-0.08% of bandwidth
- SSM uses 0.006% of compute
- Same arithmetic intensity

Two things in there are worth keeping.

The arithmetic intensity numbers are properties of the algorithms, not of our code. SSMs genuinely do a few FLOPs per byte moved; attention genuinely does hundreds. That part of the mismatch argument survives.

The near-zero utilization on both sides is also real, but it says something different than we first claimed. It says neither implementation is anywhere near the hardware's limits at batch size 1. Attention looks better only because one big matmul amortizes its launch cost, while the SSM loop pays it thousands of times.

What does not survive: the claim that the gap widens on better hardware and therefore proves an architectural mismatch. Average SSM latency is identical on both GPUs (319ms), which is exactly what you expect when launch overhead dominates. A workload that never touches the memory system will not get faster when you add bandwidth, on any GPU. That's a property of our implementation, not a verdict on silicon.

## What would change our mind

We set this up as a go/no-go test. If the gap against strong baselines is small (< 40% headroom below roofline), we pivot to software optimization or a different workload. If it's large (> 60% headroom), custom silicon is justified.

We have not run that test yet. Against naive baselines the gap is 100-1,500x, and that number is real, but it answers a question we didn't ask.

We'd also reconsider if:
- SSMs didn't match transformer quality at scale (they mostly do)
- Hybrid models dominated and attention was >50% of compute (it isn't, yet)
- A competitor was already doing this well (we haven't found one)

## What happens next

The go/no-go decision waits on four things:

1. **FlashAttention-2** for the attention side, at matching precision
2. **Official Mamba CUDA kernels** (`mamba-ssm`) for the SSM side, both recurrent and scan
3. **A fused Mamba block** (`mamba_ssm.modules.mamba_simple.Mamba`) for a realistic layer-level comparison
4. **CUDA graph capture** for decode, so launch overhead stops being the thing we measure

Then energy telemetry via pynvml, because throughput-per-watt is the actual claim and we can't make it without a watt.

## Known SSM weaknesses

SSMs aren't universally better than transformers. NVIDIA research shows they lag on:
- In-context learning (picking up new patterns from examples in the prompt)
- Copying and lookup (precise token-level recall over long contexts)

These are real tradeoffs, not bugs. They shape what we're building:
- SSU-1 is for workloads where SSMs are strong: long-context understanding, streaming inference, efficient decode
- Hybrid models (SSM + attention, like Jamba) are in scope. The custom processor handles both, but the SSM path is the one we're optimizing.
- We're not trying to make SSMs do things they fundamentally can't do well.

## Where the build stands

The benchmark side is honest about its limits now. The hardware side needs the same treatment.

1. ✅ ssm-bench runs and publishes reference implementation benchmarks (done, Oct 10, 2026)
2. 🔄 ssm-bench adds strong baselines (in progress)
3. ⏳ GPU gap report with a real go/no-go decision (blocked on 2)
4. 🔄 kalman architecture, compiler, and RTL (code written, not yet fully reviewed, tested, or error-free, so not claimed as complete)
5. ⏳ kalman prototypes on FPGA
6. ⏳ kalman validates the claim with measured throughput-per-watt

Steps 4 through 6 do not depend on step 3 finishing, so the design work continues. But if step 3 comes back negative, the design work stops. That's the deal we made with ourselves.

## Reproducing this

All the data is in this repo.

```bash
git clone https://github.com/Continum-Computing/ssm-bench.git
cd ssm-bench
make setup
make test          # correctness tests
python profiling/run_benchmarks.py   # GPU profiling, needs an NVIDIA GPU
```

Raw results:
- [Tesla T4](profiling/results/t4-colab-2026-10-10/)
- [A100](profiling/results/a100-modal-2026-10-10/)
- [T4 vs A100 comparison](profiling/results/analysis/gpu_comparison_t4_vs_a100.md)

## What this isn't

This isn't a business plan. It doesn't address market size, funding, or go-to-market.

This isn't an architecture spec. It doesn't describe the custom processor design in detail.

This isn't a benchmark methodology paper. See the main README for methodology details.

And it isn't a result. It's a statement of what we believe, what we've measured so far, what those measurements do and don't support, and what we're doing next.
