# Unthresholded Perceptron and Delta Rule

**Topic:** Perceptron and Gradient Descent

It is not possible to build a complex neural network using a thresholded perceptron because the step function prevents calculating how to update intermediate neurons[cite: 4]. To enable gradient descent, we must remove the thresholding to make the output differentiable with respect to the input[cite: 3]. 

The unthresholded perceptron simply outputs the dot product: $o(\vec{x})=\vec{w}\cdot\vec{x}$[cite: 4]. This allows us to define a differentiable cost function based on squared error: $E(\vec{w})=\frac{1}{2}\sum_{d\in D}(t_d-o_d)^2$[cite: 4]. We can then compute the gradient of the cost function, $\nabla E$, and optimize the weights using the Delta Rule (gradient descent)[cite: 4].

### Example
Instead of outputting exactly `1` or `-1`, an unthresholded perceptron outputting `0.8` for a target of `1.0` has an error of `0.2`. The squared error cost function provides a smooth, bowl-shaped landscape. Gradient descent calculates the slope of this bowl at the current weight position and takes a step downward to minimize the error.

### Code Keywords Explanation
*   `np.mean(errors ** 2)`: Computes the Mean Squared Error (MSE) cost over the batch.
*   `gradient = -np.dot(X.T, errors) / N`: Derivation of the gradient of the MSE cost function for batch updates.
*   `np.random.permutation`: Used to shuffle the dataset for the stochastic/incremental gradient descent variant.

## To implement
- [x] Linear unit and squared-error cost
- [x] Batch gradient descent using the gradient of E
- [x] Stochastic / incremental variant
- [x] Plot cost vs iterations