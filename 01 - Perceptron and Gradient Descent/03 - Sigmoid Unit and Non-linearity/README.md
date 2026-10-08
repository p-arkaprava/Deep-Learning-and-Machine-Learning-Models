# Sigmoid Unit and Non-linearity

**Topic:** Perceptron and Gradient Descent

A network composed strictly of linear neurons will mathematically collapse; it has the exact same effect as a single linear neuron[cite: 4]. To build networks capable of learning complex representations, we need a continuous and differentiable non-linear function that simulates the activation of a neuron[cite: 4]. 

This activation must behave like a step function while remaining smooth for backpropagation[cite: 4]. The standard solution is the Sigmoid function: $\sigma(x)=\frac{1}{1+e^{-x}}=y$[cite: 4]. Its derivative, required for calculating gradients, is computationally efficient: $\frac{\partial(\sigma(x))}{\partial x}=y(1-y)$[cite: 4].

### Example
If you stack two linear layers with weights $W_1$ and $W_2$, the operation is $y = W_2(W_1 x)$. Because matrix multiplication is associative, this is equal to $y = (W_2 W_1) x$. We can replace $W_2 W_1$ with a single matrix $W_3$, proving the network collapsed into a single layer. Passing the output of $W_1$ through a sigmoid function breaks this linearity, allowing the network to build hierarchical features.

### Code Keywords Explanation
*   `1 / (1 + np.exp(-x))`: The NumPy vectorization of the sigmoid mathematical formula.
*   `y * (1 - y)`: The simplified derivative of the sigmoid function, computed directly from the unit's output.
*   `np.allclose`: Verifies that the outputs of stacked linear layers equal the output of a mathematically combined single layer.

## To implement
- [x] Sigmoid and its derivative y(1 - y)
- [x] Demonstrate that stacked linear layers equal a single layer
- [x] Train a single sigmoid neuron