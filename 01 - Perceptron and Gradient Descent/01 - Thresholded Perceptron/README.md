# Thresholded Perceptron

**Topic:** Perceptron and Gradient Descent

The human brain processes information using approximately $10^{11}$ neurons, each with $10^{4}$ connections and a switching time of $10^{-3}$ seconds[cite: 3]. This system relies on distributed representation and parallel processing rather than explicit algorithmic coding[cite: 3]. 

The thresholded perceptron is a simplified mathematical model of a single neuron. It calculates a weighted sum of inputs, $net=\sum_{i=0}^{3}w_ix_i$, and applies a step function where the output $y=1$ if $net>0$, and $-1$ otherwise[cite: 3]. If the provided data is linearly separable, the perceptron learning rule will successfully find the separating hyperplane[cite: 3]. The weights are updated using the rule $\Delta w_i=\eta(t-o)x_i$, followed by $W_i=w_i+\Delta w_i$[cite: 3]. 

### Example
For a logical AND gate, the inputs $(1, 1)$ yield $1$, while $(1, 0)$, $(0, 1)$, and $(0, 0)$ yield $-1$. A single perceptron can easily draw a straight line separating the positive class from the negative classes. However, it fails on an XOR gate because no single straight line can separate the true and false outputs.

### Code Keywords Explanation
*   `np.dot(X, w)`: Computes the dot product representing the weighted sum of inputs.
*   `np.where(condition, x, y)`: Applies the step threshold, returning `x` if the condition is met, otherwise `y`.
*   `plt.contourf`: Plots the decision boundary (hyperplane) across the 2D input space.

## To implement
- [x] Perceptron with bias term (x0 = 1)
- [x] Perceptron rule: delta_w = eta * (t - o) * x
- [x] Train on AND / OR and plot the separating hyperplane
- [x] Show failure on non-separable data (XOR)