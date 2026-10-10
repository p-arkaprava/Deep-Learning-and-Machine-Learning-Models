# 01 · Architecture and Parameter Counting

How big is a feed-forward network? This module teaches you to read a layer-size list like `[784, 128, 64, 10]` and compute its **parameters, memory, FLOPs and stored activations** in your head and in code.

## Key ideas
- A dense layer with `n_in` inputs and `n_out` outputs has `n_in * n_out` weights + `n_out` biases.
- Total parameters = sum over layers. For `[784, 128, 64, 10]` that is **109,386**, and the first layer alone holds ~92 % of them.
- Forward cost ≈ `2 * (number of weights)` FLOPs per sample; stored activations ≈ `sum(n_l)` per sample.
- Width vs depth: very different parameter budgets for the same input/output sizes.

## Contents
| file | description |
|---|---|
| `01_architecture_and_parameter_counting.ipynb` | parameter counter, verification with real arrays, per-layer plots, width-vs-depth study, FLOPs, budget solver |

## Run
```bash
pip install numpy matplotlib jupyter
jupyter notebook 01_architecture_and_parameter_counting.ipynb
```

## Exercises
1. Count `[3072, 512, 256, 100]` by hand, then verify.
2. How many extra parameters does batch-norm add to each hidden layer?
3. Find the hidden width for a 5 M-parameter, 3-hidden-layer MLP.

**Next:** [02 - Matrix-Form Forward Pass](../02%20-%20Matrix-Form%20Forward%20Pass/)
