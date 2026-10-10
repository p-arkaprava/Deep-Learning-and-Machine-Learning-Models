# 06 · Memory Footprint and Multi-GPU Training

Estimate whether a model fits in GPU memory *before* running it, and see how data parallelism splits the work.

## Key ideas
- fp32 + Adam costs **16 bytes per parameter** (4 weights + 4 grads + 8 Adam states), plus activations that grow with batch size.
- A 7B model needs ~112 GB for weights, grads and Adam states alone - more than one 80 GB GPU.
- **Data parallel**: full model copy per GPU, different data slice, gradients averaged by all-reduce - mathematically identical to one big batch.
- **Ring all-reduce** sends only `2(K-1)/K` of the gradient size per GPU.
- **ZeRO / FSDP** shard optimizer states, gradients and weights across GPUs; **pipeline parallelism** splits by layers.

## Contents
| file | description |
|---|---|
| `06_memory_footprint_and_multi_gpu.ipynb` | memory estimator verified against real arrays, batch-size scaling, LLM-scale table, ZeRO stages, simulated data parallelism, ring all-reduce, pipeline split |

## Run
```bash
jupyter notebook 06_memory_footprint_and_multi_gpu.ipynb
```

## Exercises
1. Add gradient checkpointing to the estimator.
2. Add mixed-precision (bf16) accounting.
3. Prove gradient accumulation over 4 micro-batches equals one big batch.

**Previous:** [05](../05%20-%20Training%20a%20Small%20Network/)
