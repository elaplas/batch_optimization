# How splitting data points in batches accelerate gradient computation 

Given $N$ number of data points $X_i$, $N$ number of measurements $Y_i$, and a MLP, that is abstracted as \( f(X_i, W) \), the following loss function is defined:

\[ \text{L} = \frac{1}{N}\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 \]

Where:
- \( n \) is the number of data points with $m$ dimensions
- \( Y_i \) is the observation for the \( i \)-th data point.
- \( \hat{Y}_i \) is the prediction for the \( i \)-th data point, which can be assumed that is the output of a multi layer perceptron, \( \hat{Y}_i = f(X_i, W) \). 

In the loss function $L$, the data points $X_i$ vary but the weights $W$ are constant. The weights should be tuned in a way that the predictions \( \hat{Y}_i\) are close enough to $Y_i$ so that $L$ goes to zero. The weights are iteratively tuned using gradient descent:

\[ W_{k+1} = W_{k} - \alpha \nabla_{W} L(W_{k}) \]

- \( W \) represents the parameters (weights) of the model.
- \( \alpha \) is the learning rate, a positive scalar that controls the step size.
- \( \nabla_{W} L(W) \) is the gradient of the function \( f \) with respect to \( W \) at \( W_k \) .

The gradient of loss function $L$ with respect to weights $W$ is calculated using gradient descent. The number of data points could be too many leading to a very big loss function and consequently too much sequential computation at the time of gradient computation. Therefore the data points are split in batches so that they are independently substituted in loss function and the corresponding gradient is calculated separately. This makes loss function smaller and enables parallel tunning of weights.

This python implementation shows the sequential nature of computation of gradients in a dataset and how batches can help parallelizing the computation of gradients by splitting data points.   