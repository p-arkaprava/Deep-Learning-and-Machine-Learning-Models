# Perceptron and Gradient Descent
From a single neuron to gradient-based learning[cite: 1].

The human brain processes information using approximately $10^{11}$ neurons, with $10^{4}$ connections per neuron and a switching time of $10^{-3}$ seconds[cite: 3]. It relies on distributed representation and parallel processing rather than explicit algorithmic coding[cite: 3]. This module traces the mathematical evolution from a simplified model of a single biological neuron to the differentiable units that make deep learning possible.

### 1. Thresholded Perceptron
The thresholded perceptron calculates a weighted sum of its inputs ($net=\sum w_ix_i$) and applies a hard step function to produce a binary output[cite: 3]. If the dataset is linearly separable, it successfully finds the separating hyperplane using the update rule $\Delta w_i=\eta(t-o)x_i$[cite: 3]. However, this model cannot be used to build complex neural networks because the hard threshold prevents us from calculating how to update intermediate hidden neurons[cite: 4].

### 2. Unthresholded Perceptron and Delta Rule
To train networks effectively, we must get rid of the hard thresholding to make the output differentiable with respect to the input[cite: 3]. The unthresholded perceptron simply outputs the dot product of the weights and inputs: $o(\vec{x})=\vec{w}\cdot\vec{x}$[cite: 4]. This allows us to define a differentiable cost function based on squared error, $E(\vec{w})=\frac{1}{2}\sum_{d\in D}(t_d-o_d)^2$, and use gradient descent ($\nabla E$) to optimize the weights[cite: 4].

### 3. Sigmoid Unit and Non-linearity
A network composed solely of linear neurons mathematically collapses and has the exact same effect as a single linear neuron[cite: 4]. To enable the network to learn complex, non-linear relationships, we need an activation function that is continuous, differentiable, and simulates the firing of a neuron[cite: 4]. The sigmoid function, $\sigma(x)=\frac{1}{1+e^{-x}}=y$, provides this essential non-linearity, and its derivative is efficiently calculated as $y(1-y)$[cite: 4]. 

## Topics
- [x] 01 - Thresholded Perceptron[cite: 1]
- [x] 02 - Unthresholded Perceptron and Delta Rule[cite: 1]
- [x] 03 - Sigmoid Unit and Non-linearity[cite: 1]