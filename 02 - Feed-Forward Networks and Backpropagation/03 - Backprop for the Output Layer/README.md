# 03 · Backprop for the Output Layer

Derive and implement the gradient of the loss with respect to the last layer, and prove it with a numerical gradient check.

## Key ideas
- Error signal: `delta = dL/dZ`. Then `dW = A_prev.T @ delta`, `db = delta.sum(axis=0)`, `dA_prev = delta @ W.T`.
- Softmax + cross-entropy collapses to **`delta = (P - Y) / N`**.
- Sigmoid + binary cross-entropy and identity + MSE give the same simple form; sigmoid + MSE adds a `a(1-a)` factor that **kills the gradient** when the neuron saturates.
- **Gradient checking** with central differences (`relative error < 1e-6`) catches almost every backprop bug.

## Contents
| file | description |
|---|---|
| `03_backprop_output_layer.ipynb` | Jacobian derivation check, output-layer forward/backward, `numerical_grad`, BCE vs MSE comparison, saturation plot |

## Run
```bash
jupyter notebook 03_backprop_output_layer.ipynb
```

## Exercises
1. Add L2 regularisation and derive the new `dW`.
2. Implement label smoothing and derive its `delta`.
3. Break the gradient on purpose and watch the check fail.

**Previous:** [02](../02%20-%20Matrix-Form%20Forward%20Pass/) · **Next:** [04 - Backprop for Hidden Layers](../04%20-%20Backprop%20for%20Hidden%20Layers/)
