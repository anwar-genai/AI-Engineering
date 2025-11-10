"""
WEEK 3 - DAY 1-2: PyTorch Tensors & Autograd
============================================
Learn the fundamentals of PyTorch: tensors and automatic differentiation


Topics:
- Tensor creation and operations
- NumPy vs PyTorch tensors
- Autograd (automatic differentiation)
- Gradient computation
- Building blocks for neural networks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("WEEK 3 - DAY 1-2: PyTorch Tensors & Autograd")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================
# PART 1: Tensor Basics
# ============================================
print("\n>>> PART 1: Tensor Creation and Operations")

# Create tensors (similar to NumPy arrays)
print("\n--- Creating Tensors ---")

# From Python lists
t1 = torch.tensor([1, 2, 3, 4, 5])
print(f"From list: {t1}")
print(f"Shape: {t1.shape}, dtype: {t1.dtype}")

# 2D tensor (matrix)
t2 = torch.tensor([[1, 2, 3], 
                   [4, 5, 6]])
print(f"\n2D tensor:\n{t2}")
print(f"Shape: {t2.shape}")

# Common initialization methods
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
random = torch.randn(3, 3)  # Normal distribution
uniform = torch.rand(2, 2)  # Uniform [0, 1)
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)

print(f"\nZeros:\n{zeros}")
print(f"\nOnes:\n{ones}")
print(f"\nRandom (normal):\n{random}")
print(f"\nRandom (uniform):\n{uniform}")
print(f"\nArange: {arange}")
print(f"\nLinspace: {linspace}")

# Tensor with specific dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"\nFloat tensor: {float_tensor}, dtype: {float_tensor.dtype}")
print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")

# ============================================
# PART 2: Tensor Operations
# ============================================
print("\n>>> PART 2: Tensor Operations")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"a ** 2 = {a ** 2}")

# Matrix operations
A = torch.tensor([[1.0, 2.0], 
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], 
                  [7.0, 8.0]])

print(f"\nMatrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Matrix multiplication
C = torch.matmul(A, B)  # or A @ B
print(f"\nA @ B =\n{C}")

# Transpose
print(f"\nA transpose:\n{A.T}")

# Aggregations
print(f"\nSum: {a.sum()}")
print(f"Mean: {a.mean()}")
print(f"Max: {a.max()}")
print(f"Min: {a.min()}")
print(f"Std: {a.std()}")

# Reshaping
x = torch.arange(12)
print(f"\nOriginal: {x}")
print(f"Shape: {x.shape}")

x_reshaped = x.reshape(3, 4)
print(f"\nReshaped (3, 4):\n{x_reshaped}")

x_reshaped2 = x.view(2, 6)  # view is like reshape but faster
print(f"\nView (2, 6):\n{x_reshaped2}")

# ============================================
# PART 3: NumPy Bridge
# ============================================
print("\n>>> PART 3: NumPy ↔ PyTorch Conversion")

# NumPy to PyTorch
np_array = np.array([1, 2, 3, 4, 5])
torch_from_numpy = torch.from_numpy(np_array)
print(f"\nNumPy array: {np_array}")
print(f"Torch tensor: {torch_from_numpy}")

# PyTorch to NumPy
torch_tensor = torch.tensor([6, 7, 8, 9, 10])
numpy_from_torch = torch_tensor.numpy()
print(f"\nTorch tensor: {torch_tensor}")
print(f"NumPy array: {numpy_from_torch}")

# IMPORTANT: They share memory!
np_array[0] = 100
print(f"\nAfter modifying NumPy array:")
print(f"NumPy: {np_array}")
print(f"Torch: {torch_from_numpy}")  # Also changed!

# ============================================
# PART 4: Indexing and Slicing
# ============================================
print("\n>>> PART 4: Indexing and Slicing")

x = torch.arange(20).reshape(4, 5)
print(f"\nTensor:\n{x}")

# Basic indexing
print(f"\nFirst row: {x[0]}")
print(f"First column: {x[:, 0]}")
print(f"Element (2, 3): {x[2, 3]}")

# Slicing
print(f"\nFirst 2 rows:\n{x[:2]}")
print(f"Last 2 columns:\n{x[:, -2:]}")

# Boolean indexing
mask = x > 10
print(f"\nMask (x > 10):\n{mask}")
print(f"Values > 10: {x[mask]}")

# ============================================
# PART 5: Autograd (Automatic Differentiation)
# ============================================
print("\n>>> PART 5: Autograd - The Magic of PyTorch!")

print("\n--- Simple Example: f(x) = x² ---")

# Create tensor with gradient tracking
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad: {x.requires_grad}")

# Forward pass: compute function
y = x ** 2
print(f"\ny = x² = {y}")

# Backward pass: compute gradient dy/dx
y.backward()
print(f"\ndy/dx = 2x = {x.grad}")  # Should be 2*3 = 6

# ============================================
print("\n--- More Complex Example: f(x) = 3x² + 2x + 1 ---")

x = torch.tensor(2.0, requires_grad=True)
y = 3 * x**2 + 2 * x + 1

print(f"x = {x}")
print(f"y = 3x² + 2x + 1 = {y}")

y.backward()
print(f"\ndy/dx = 6x + 2 = {x.grad}")  # Should be 6*2 + 2 = 14

# ============================================
print("\n--- Multi-variable: f(x, y) = x²y + y³ ---")

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x**2 * y + y**3
print(f"\nx = {x}, y = {y}")
print(f"z = x²y + y³ = {z}")

z.backward()
print(f"\n∂z/∂x = 2xy = {x.grad}")  # Should be 2*2*3 = 12
print(f"∂z/∂y = x² + 3y² = {y.grad}")  # Should be 4 + 27 = 31

# ============================================
print("\n--- Vector Example ---")

# Input vector
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Operation
y = x ** 2
z = y.sum()  # Scalar output needed for backward()

print(f"\nx = {x}")
print(f"y = x² = {y}")
print(f"z = sum(y) = {z}")

z.backward()
print(f"\ndz/dx = 2x = {x.grad}")  # [2, 4, 6]

# ============================================
# PART 6: Gradient Descent Example
# ============================================
print("\n>>> PART 6: Gradient Descent from Scratch")

print("\n--- Minimize f(x) = (x - 3)² ---")
print("Optimal x should be 3")

# Initialize
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1
iterations = 50

# Track history
x_history = []
loss_history = []

print("\nIteration | x      | Loss   | Gradient")
print("-" * 45)

for i in range(iterations):
    # Forward pass
    loss = (x - 3) ** 2
    
    # Backward pass
    loss.backward()
    
    # Record
    x_history.append(x.item())
    loss_history.append(loss.item())
    
    if i % 10 == 0:
        print(f"{i:9d} | {x.item():6.3f} | {loss.item():6.3f} | {x.grad.item():8.3f}")
    
    # Update parameters (gradient descent)
    with torch.no_grad():  # Don't track these operations
        x -= learning_rate * x.grad
    
    # Zero gradients for next iteration
    x.grad.zero_()

print(f"\nFinal x: {x.item():.6f}")
print(f"Target x: 3.0")
print(f"Error: {abs(x.item() - 3.0):.6f}")

# ============================================
# PART 7: Visualizations
# ============================================
print("\n>>> PART 7: Visualizing Gradient Descent")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curve
ax1 = axes[0]
ax1.plot(loss_history, linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs Iteration')
ax1.grid(True, alpha=0.3)

# Plot 2: Parameter convergence
ax2 = axes[1]
ax2.plot(x_history, linewidth=2, label='x')
ax2.axhline(y=3.0, color='r', linestyle='--', label='Target (x=3)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('x value')
ax2.set_title('Parameter Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day1_gradient_descent.png', dpi=150)
print("\n✅ Visualization saved: day1_gradient_descent.png")
plt.close()

# ============================================
# PART 8: Linear Regression with Autograd
# ============================================
print("\n>>> PART 8: Linear Regression Example")

# Generate synthetic data: y = 2x + 1 + noise
torch.manual_seed(42)
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y_true = 2 * X + 1 + torch.randn(100, 1) * 0.5

# Initialize parameters
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
epochs = 100

print("\nTraining linear regression: y = wx + b")
print(f"True parameters: w=2, b=1")
print(f"Initial parameters: w={w.item():.3f}, b={b.item():.3f}")

for epoch in range(epochs):
    # Forward pass
    y_pred = X @ w + b
    
    # Compute loss (MSE)
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | w: {w.item():.3f} | b: {b.item():.3f}")

print(f"\nFinal parameters: w={w.item():.3f}, b={b.item():.3f}")
print(f"True parameters:  w=2.000, b=1.000")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X.numpy(), y_true.numpy(), alpha=0.5, label='Data')
plt.plot(X.numpy(), (X @ w + b).detach().numpy(), 'r-', linewidth=2, label='Fitted line')
plt.plot(X.numpy(), (2 * X + 1).numpy(), 'g--', linewidth=2, label='True line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Autograd')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day1_linear_regression.png', dpi=150)
print("✅ Visualization saved: day1_linear_regression.png")
plt.close()

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. TENSORS:
   - PyTorch's version of NumPy arrays
   - Can run on GPU (we'll do this later)
   - Similar operations to NumPy

2. AUTOGRAD:
   - Automatic differentiation
   - Tracks operations for backpropagation
   - Set requires_grad=True to track gradients
   - Call .backward() to compute gradients

3. GRADIENT DESCENT:
   - Update parameters: x = x - lr * gradient
   - Zero gradients after each update
   - Use torch.no_grad() when updating

4. WORKFLOW:
   - Forward pass: compute output
   - Compute loss
   - Backward pass: loss.backward()
   - Update parameters
   - Zero gradients

5. WHY PYTORCH?
   - Automatic gradients (no manual calculus!)
   - GPU acceleration (coming soon)
   - Dynamic computation graphs
   - Pythonic and intuitive

NEXT: Build neural networks with nn.Module!
""")
print("="*70)

print("\n✅ Day 1-2 Complete! Move to day3_neural_networks.py")