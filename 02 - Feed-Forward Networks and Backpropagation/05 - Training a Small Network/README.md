# 05 · Training a Small Network

Combine forward + backward + an optimiser and train a real classifier from scratch in NumPy.

## Key ideas
- Loop: shuffle, take a mini-batch, forward, loss, backward, update `theta <- theta - lr * grad`.
- Momentum: `v <- mu * v - lr * g; theta <- theta + v`.
- A hidden layer is what lets the network solve XOR and the 3-arm spiral (100 % accuracy with a 2-64-64-3 net).
- **Learning rate** is the most sensitive hyper-parameter: too small = slow, too large = divergence.

## Contents
| file | description |
|---|---|
| `05_training_a_small_network.ipynb` | `MLP` class (forward/backward/SGD+momentum/fit), XOR warm-up, spiral dataset, loss/accuracy curves, decision regions, learning-rate sweep |

## Run
```bash
jupyter notebook 05_training_a_small_network.ipynb
```

## Exercises
1. Compare `momentum=0` and `0.9`.
2. Add weight decay to `step`.
3. Swap ReLU for tanh in both `forward` and `backward`.
4. Train on `sklearn.datasets.load_digits` with `sizes=[64, 64, 10]`.

**Previous:** [04](../04%20-%20Backprop%20for%20Hidden%20Layers/) · **Next:** [06 - Memory Footprint and Multi-GPU Training](../06%20-%20Memory%20Footprint%20and%20Multi-GPU%20Training/)
