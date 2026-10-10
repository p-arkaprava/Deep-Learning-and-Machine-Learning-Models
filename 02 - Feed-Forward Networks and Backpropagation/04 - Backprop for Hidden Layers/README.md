# 04 · Backprop for Hidden Layers

Send the error signal backwards through any number of layers.

## Key ideas
- Recursion: **`delta[l] = (delta[l+1] @ W[l+1].T) * g'(Z[l])`**.
- Every layer: `dW[l] = A[l-1].T @ delta[l]`, `db[l] = delta[l].sum(axis=0)`.
- Backprop is reverse-mode autodiff on a chain: one forward sweep (store `Z`, `A`), one backward sweep. Backward costs about 2x forward.
- Each step multiplies by `g'`; sigmoid (`g' <= 0.25`) makes gradients **vanish**, ReLU + He initialisation keeps them healthy.

## Contents
| file | description |
|---|---|
| `04_backprop_hidden_layers.ipynb` | general L-layer forward/backward, shape trace, full gradient check on a 4-layer net, vanishing-gradient experiment (sigmoid vs tanh vs ReLU) |

## Run
```bash
jupyter notebook 04_backprop_hidden_layers.ipynb
```

## Exercises
1. Add `leaky_relu` and gradient-check it.
2. Plot `||delta||` per layer for the sigmoid network.
3. Increase depth to 30 - when does ReLU + He start exploding?

**Previous:** [03](../03%20-%20Backprop%20for%20the%20Output%20Layer/) · **Next:** [05 - Training a Small Network](../05%20-%20Training%20a%20Small%20Network/)
