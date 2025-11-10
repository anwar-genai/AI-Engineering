"""
WEEK 3 - DAY 3-4: Building Neural Networks
==========================================
Learn to build neural networks with PyTorch nn.Module


Topics:
- nn.Module and layers
- Activation functions
- Forward pass
- Loss functions
- Optimizers
- Training loop
- MNIST digit classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("WEEK 3 - DAY 3-4: Neural Networks with PyTorch")
print("="*70)

# ============================================
# PART 1: Understanding Layers
# ============================================
print("\n>>> PART 1: Neural Network Layers")

# Linear layer (fully connected)
linear = nn.Linear(in_features=10, out_features=5)
print("\n--- Linear Layer ---")
print(f"Layer: {linear}")
print(f"Weight shape: {linear.weight.shape}")  # (out, in)
print(f"Bias shape: {linear.bias.shape}")

# Test with random input
x = torch.randn(3, 10)  # Batch of 3 samples, 10 features each
output = linear(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")  # (3, 5)

# Common activation functions
print("\n--- Activation Functions ---")
x = torch.linspace(-3, 3, 100)

relu = nn.ReLU()(x)
sigmoid = nn.Sigmoid()(x)
tanh = nn.Tanh()(x)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x.numpy(), relu.numpy())
axes[0].set_title('ReLU')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x.numpy(), sigmoid.numpy())
axes[1].set_title('Sigmoid')
axes[1].grid(True, alpha=0.3)

axes[2].plot(x.numpy(), tanh.numpy())
axes[2].set_title('Tanh')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day3_activations.png', dpi=150)
print("✅ Activation functions plotted")
plt.close()

# ============================================
# PART 2: Building a Simple Neural Network
# ============================================
print("\n>>> PART 2: Building Neural Network Class")

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define forward pass
        x = self.fc1(x)      # Linear transformation
        x = self.relu(x)     # Activation
        x = self.fc2(x)      # Output layer
        return x

# Create model
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
print("\nModel architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test forward pass
x = torch.randn(32, 784)  # Batch of 32 images (28x28 flattened)
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")  # (32, 10) - 10 classes

# ============================================
# PART 3: Loss Functions and Optimizers
# ============================================
print("\n>>> PART 3: Loss Functions and Optimizers")

# Common loss functions
print("\n--- Loss Functions ---")

# For classification
criterion_ce = nn.CrossEntropyLoss()
print(f"Cross Entropy Loss: {criterion_ce}")

# For regression
criterion_mse = nn.MSELoss()
print(f"MSE Loss: {criterion_mse}")

# Optimizers
print("\n--- Optimizers ---")

# SGD (Stochastic Gradient Descent)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
print(f"SGD: {optimizer_sgd}")

# Adam (usually better)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
print(f"Adam: {optimizer_adam}")

# ============================================
# PART 4: Training Loop Template
# ============================================
print("\n>>> PART 4: Understanding the Training Loop")

print("""
STANDARD PYTORCH TRAINING LOOP:
================================

for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        
        # 1. Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # 2. Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        
        # 3. Update parameters
        optimizer.step()       # Update weights
        
    # Print progress
    print(f'Epoch {epoch}, Loss: {loss.item()}')
""")

# ============================================
# PART 5: Toy Example - XOR Problem
# ============================================
print("\n>>> PART 5: Solving XOR Problem")

# XOR dataset
X_xor = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]])

y_xor = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]])

print("\nXOR Truth Table:")
print("Input  | Output")
print("-------|-------")
for x, y in zip(X_xor, y_xor):
    print(f"{x.numpy()} | {y.item()}")

# Build model for XOR
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 2 inputs, 4 hidden
        self.fc2 = nn.Linear(4, 1)  # 4 hidden, 1 output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize
xor_model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(xor_model.parameters(), lr=0.01)

# Train
print("\nTraining XOR network...")
losses = []

for epoch in range(2000):
    # Forward pass
    outputs = xor_model(X_xor)
    loss = criterion(outputs, y_xor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")

# Test
print("\nTrained XOR predictions:")
with torch.no_grad():
    predictions = xor_model(X_xor)
    print("Input  | Target | Predicted | Rounded")
    print("-------|--------|-----------|--------")
    for x, y_true, y_pred in zip(X_xor, y_xor, predictions):
        print(f"{x.numpy()} | {y_true.item():.1f}    | {y_pred.item():.4f}    | {round(y_pred.item())}")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Problem - Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day3_xor_loss.png', dpi=150)
print("\n✅ XOR training plot saved")
plt.close()

# ============================================
# PART 6: MNIST Digit Classification
# ============================================
print("\n>>> PART 6: MNIST Digit Classification")

# Load MNIST dataset
print("\nDownloading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST(
    'D:/ai_engineering/datasets', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.MNIST(
    'D:/ai_engineering/datasets', 
    train=False, 
    download=True, 
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Visualize some samples
print("\n✅ Visualizing MNIST samples...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day3_mnist_samples.png', dpi=150)
print("✅ MNIST samples saved")
plt.close()

# ============================================
# PART 7: Build MNIST Classifier
# ============================================
print("\n>>> PART 7: Building MNIST Classifier")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten image
        x = x.view(-1, 28*28)
        
        # Layer 1
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Layer 2
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        return x

# Initialize model
mnist_model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)

print("\nModel architecture:")
print(mnist_model)
print(f"\nTotal parameters: {sum(p.numel() for p in mnist_model.parameters()):,}")

# ============================================
# PART 8: Training MNIST Model
# ============================================
print("\n>>> PART 8: Training MNIST Model")

num_epochs = 5
train_losses = []
test_accuracies = []

print("\nEpoch | Train Loss | Test Accuracy | Time")
print("-" * 50)

import time

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training
    mnist_model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = mnist_model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Testing
    mnist_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = mnist_model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    
    elapsed = time.time() - start_time
    print(f"{epoch+1:5d} | {avg_loss:10.4f} | {accuracy:13.2f}% | {elapsed:.1f}s")

print(f"\n✅ Training complete!")
print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")

# ============================================
# PART 9: Evaluation and Visualization
# ============================================
print("\n>>> PART 9: Model Evaluation")

# Plot training progress
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_accuracies, 'g-', linewidth=2, marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Test Accuracy')
axes[1].set_ylim([90, 100])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day3_mnist_training.png', dpi=150)
print("✅ Training plots saved")
plt.close()

# Test on individual samples
print("\n--- Testing on individual digits ---")
mnist_model.eval()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        image, true_label = test_dataset[i]
        output = mnist_model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        
        ax.imshow(image.squeeze(), cmap='gray')
        color = 'green' if predicted.item() == true_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {predicted.item()}', color=color)
        ax.axis('off')

plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day3_mnist_predictions.png', dpi=150)
print("✅ Prediction samples saved")
plt.close()

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds = []
all_labels = []

mnist_model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = mnist_model(data)
        _, predicted = torch.max(output, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(target.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('MNIST Confusion Matrix')
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day3_mnist_confusion.png', dpi=150)
print("✅ Confusion matrix saved")
plt.close()

# ============================================
# PART 10: Saving and Loading Model
# ============================================
print("\n>>> PART 10: Saving Model")

# Save model
torch.save(mnist_model.state_dict(), 'D:/ai_engineering/week3_pytorch_basics/mnist_model.pth')
print("✅ Model saved: mnist_model.pth")

# Load model (example)
loaded_model = MNISTNet()
loaded_model.load_state_dict(torch.load('D:/ai_engineering/week3_pytorch_basics/mnist_model.pth'))
loaded_model.eval()
print("✅ Model loaded successfully")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. NN.MODULE:
   - Base class for all neural networks
   - Define layers in __init__
   - Define forward pass in forward()
   
2. TRAINING LOOP:
   - Forward pass → Compute loss
   - Backward pass → loss.backward()
   - Optimizer step → optimizer.step()
   - Zero gradients → optimizer.zero_grad()

3. LAYERS:
   - nn.Linear: Fully connected layer
   - nn.ReLU, nn.Sigmoid: Activation functions
   - nn.Dropout: Regularization

4. LOSS FUNCTIONS:
   - CrossEntropyLoss: Classification
   - MSELoss: Regression
   - BCELoss: Binary classification

5. OPTIMIZERS:
   - SGD: Simple, sometimes slow
   - Adam: Usually best choice
   - Learning rate matters!

6. DATA LOADING:
   - Use DataLoader for batching
   - Shuffle training data
   - Don't shuffle test data

7. EVALUATION:
   - model.eval() before testing
   - torch.no_grad() to save memory
   - Track metrics (accuracy, loss)

MNIST RESULTS:
- Training samples: 60,000
- Test samples: 10,000
- Final accuracy: ~97-98%
- Parameters: ~100,000

NEXT: Convolutional Neural Networks (CNNs)!
""")
print("="*70)

print("\n✅ Day 3-4 Complete! Move to day5_cnn.py")