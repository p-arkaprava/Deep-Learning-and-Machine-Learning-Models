# 02 · Matrix-Form Forward Pass

Turn "a neuron computes a weighted sum" into one matrix expression that handles a whole batch at once.

## Key ideas
- Row-major convention: `Z = A_prev @ W + b`, with shapes `(N, n_in) @ (n_in, n_out) + (n_out,) -> (N, n_out)`.
- Activations (sigmoid, tanh, ReLU) add the non-linearity; softmax turns the last layer into probabilities.
- Always **subtract the row maximum** in softmax - otherwise `exp` overflows.
- The vectorised version is typically 1000x+ faster than Python loops.
- Cache every `Z` and `A`: backprop needs them.

## Contents
| file | description |
|---|---|
| `02_matrix_form_forward_pass.ipynb` | neuron as dot product, activations + derivatives, loop-vs-matrix timing, full forward with cache and shape trace, softmax stability, hand-checkable 2-2-1 example |

## Run
```bash
jupyter notebook 02_matrix_form_forward_pass.ipynb
```

## Exercises
1. Redo the 2-2-1 example with `tanh` by hand and in code.
2. Which dimension of every tensor depends on the batch size?
3. Time `dense_loops` with only the outer sample loop replaced by `X[n] @ W + b`.

**Previous:** [01](../01%20-%20Architecture%20and%20Parameter%20Counting/) · **Next:** [03 - Backprop for the Output Layer](../03%20-%20Backprop%20for%20the%20Output%20Layer/)
