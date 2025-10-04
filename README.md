# PyTorch Autograd Demo Notebook
This Jupyter Notebook demonstrates PyTorch's automatic differentiation (Autograd) functionality, which is the core mechanism for training deep learning models.

Main Content Overview
1. Basic Autograd Demo
Create tensors that require gradients: x = torch.tensor(3.0, requires_grad=True)

Build computation graph: y = x**2, z = torch.sin(y)

Backward propagation to compute gradients: z.backward()

View gradients: x.grad

2. Manual Gradient Calculation for Binary Cross-Entropy Loss
Implement complete forward propagation for binary classification problem

Manually calculate gradients of loss function with respect to weights and bias:

dloss_dy_pred: Gradient of loss with respect to prediction

dy_pred_dz: Gradient of sigmoid function

dz_dw and dz_db: Gradients of linear part

3. Automatic Gradient Calculation Using Autograd
Set requires_grad=True to let PyTorch track computations

Automatically compute gradients: loss.backward()

Compare manual calculation results with automatic calculation results

4. Autograd with Tensor Operations
Operations on vector tensors: y = (x**2).mean()

Compute gradients of mean value

5. Gradient Management
Clear gradients: x.grad.zero_()

Three methods to disable gradient tracking:

requires_grad_(False)

detach() method

torch.no_grad() context manager

Key Concepts
How Autograd Works
Computation Graph Construction: PyTorch dynamically builds computation graphs during tensor operations

Gradient Calculation: Automatically computes derivatives using chain rule

Gradient Accumulation: Gradients accumulate and need to be manually cleared

Application Scenarios
Backpropagation in neural network training

Any optimization problem requiring gradient calculation

Model debugging and gradient checking

This notebook effectively demonstrates the transition from manual gradient calculation to automation using PyTorch Autograd, helping to understand the fundamental principles behind deep learning frameworks.

Code Execution Results Summary
Basic Example:
x = 3.0, y = x² = 9.0

After y.backward(), x.grad = 6.0 (derivative of x² is 2x)

Chain Rule Example:
x = 3.0, y = x² = 9.0, z = sin(y) = sin(9) ≈ 0.4121

After z.backward(), x.grad ≈ -5.4668 (derivative of sin(x²) is 2x·cos(x²))

Binary Classification Example:
Input x = 6.7, true label y = 0.0

Manual gradients: dl_dw = 6.6970, dl_db = 0.9995

Autograd gradients: w.grad = 6.6970, b.grad = 0.9995

Both methods produce identical results!

Vector Operations:
x = [1.0, 2.0, 4.0], y = mean(x²) = 7.0

Gradients: x.grad = [0.6667, 1.3333, 2.6667] (derivative of mean(x²) is 2x/3)

This demonstrates that PyTorch Autograd reliably computes the same gradients as manual derivation, making deep learning training much more efficient and less error-prone!



