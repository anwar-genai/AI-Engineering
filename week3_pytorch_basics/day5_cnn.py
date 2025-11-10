"""
WEEK 3 - DAY 5-7: Convolutional Neural Networks (CNNs)
=======================================================
Learn CNNs for image classification


Topics:
- Convolution layers
- Pooling layers
- CNN architecture
- CIFAR-10 classification
- Data augmentation
- Model evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

print("="*70)
print("WEEK 3 - DAY 5-7: Convolutional Neural Networks")
print("="*70)

# ============================================
# PART 1: Understanding Convolution
# ============================================
print("\n>>> PART 1: Convolution Operation")

# Create a simple image (5x5)
image = torch.tensor([[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]], dtype=torch.float32).unsqueeze(0)  # Add batch and channel dims

print("\nOriginal image (5x5):")
print(image.squeeze().numpy())

# Define a 3x3 convolutional layer
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

# Set a simple kernel (edge detection)
with torch.no_grad():
    conv.weight = nn.Parameter(torch.tensor([[
        [[-1, -1, -1],
         [ 0,  0,  0],
         [ 1,  1,  1]]
    ]], dtype=torch.float32))
    conv.bias = nn.Parameter(torch.zeros(1))

# Apply convolution
output = conv(image)
print("\nConvolution output (3x3):")
print(output.squeeze().detach().numpy())

print(f"\nInput shape: {image.shape}")
print(f"Output shape: {output.shape}")
print("Note: (5-3)/1 + 1 = 3 (formula for output size)")

# ============================================
# PART 2: CNN Layers
# ============================================
print("\n>>> PART 2: CNN Building Blocks")

print("\n--- Convolution Layer ---")
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
print(f"Conv2d: {conv_layer}")
print(f"Parameters: {sum(p.numel() for p in conv_layer.parameters()):,}")

print("\n--- Pooling Layers ---")
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
print(f"MaxPool2d: {maxpool}")
print(f"AvgPool2d: {avgpool}")

# Test pooling
x = torch.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width
out_max = maxpool(x)
out_avg = avgpool(x)
print(f"\nInput: {x.shape}")
print(f"After MaxPool: {out_max.shape}")
print(f"After AvgPool: {out_avg.shape}")

print("\n--- Batch Normalization ---")
batchnorm = nn.BatchNorm2d(16)
print(f"BatchNorm2d: {batchnorm}")

# ============================================
# PART 3: Simple CNN Architecture
# ============================================
print("\n>>> PART 3: Building a CNN")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model
simple_cnn = SimpleCNN()
print("\nSimple CNN Architecture:")
print(simple_cnn)
print(f"\nTotal parameters: {sum(p.numel() for p in simple_cnn.parameters()):,}")

# Test forward pass
test_input = torch.randn(1, 1, 28, 28)
test_output = simple_cnn(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")

# ============================================
# PART 4: Load CIFAR-10 Dataset
# ============================================
print("\n>>> PART 4: Loading CIFAR-10 Dataset")

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Data transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("\nDownloading CIFAR-10...")
train_dataset = datasets.CIFAR10(
    'D:/ai_engineering/datasets',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    'D:/ai_engineering/datasets',
    train=False,
    download=True,
    transform=transform_test
)

print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {classes}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Visualize samples
print("\n✅ Visualizing CIFAR-10 samples...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    # Denormalize
    image = image / 2 + 0.5
    image = image.permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.set_title(classes[label])
    ax.axis('off')
plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day5_cifar10_samples.png', dpi=150)
print("✅ CIFAR-10 samples saved")
plt.close()

# ============================================
# PART 5: Build CIFAR-10 CNN
# ============================================
print("\n>>> PART 5: Building CIFAR-10 CNN")

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1: 32x32 -> 16x16
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: 16x16 -> 8x8
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3: 8x8 -> 4x4
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize model
cifar_model = CIFAR10Net()
print("\nCIFAR-10 CNN Architecture:")
print(cifar_model)
print(f"\nTotal parameters: {sum(p.numel() for p in cifar_model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)

# ============================================
# PART 6: Training CIFAR-10 Model
# ============================================
print("\n>>> PART 6: Training CIFAR-10 CNN")

num_epochs = 10
train_losses = []
train_accs = []
test_accs = []

print("\nEpoch | Train Loss | Train Acc | Test Acc | Time")
print("-" * 60)

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training
    cifar_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        outputs = cifar_model(data)
        loss = criterion(outputs, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Testing
    cifar_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = cifar_model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_acc = 100 * correct / total
    test_accs.append(test_acc)
    
    elapsed = time.time() - start_time
    print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {test_acc:8.2f}% | {elapsed:.1f}s")

print(f"\n✅ Training complete!")
print(f"Best test accuracy: {max(test_accs):.2f}%")

# ============================================
# PART 7: Visualization
# ============================================
print("\n>>> PART 7: Training Visualization")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(train_losses, 'b-', linewidth=2, label='Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(train_accs, 'b-', linewidth=2, label='Train Accuracy')
axes[1].plot(test_accs, 'r-', linewidth=2, label='Test Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Train vs Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day5_cifar10_training.png', dpi=150)
print("✅ Training curves saved")
plt.close()

# ============================================
# PART 8: Model Evaluation
# ============================================
print("\n>>> PART 8: Model Evaluation")

# Per-class accuracy
class_correct = [0] * 10
class_total = [0] * 10

cifar_model.eval()
with torch.no_grad():
    for data, target in test_loader:
        outputs = cifar_model(data)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == target).squeeze()
        for i in range(len(target)):
            label = target[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print("\nPer-class accuracy:")
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:10s}: {acc:5.2f}%")

# Visualize predictions
print("\n✅ Visualizing predictions...")
cifar_model.eval()
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        image, true_label = test_dataset[i * 10]
        output = cifar_model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        
        # Denormalize image
        img = image / 2 + 0.5
        img = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img)
        color = 'green' if predicted.item() == true_label else 'red'
        ax.set_title(f'True: {classes[true_label]}\nPred: {classes[predicted.item()]}', 
                    color=color, fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('D:/ai_engineering/week3_pytorch_basics/day5_cifar10_predictions.png', dpi=150)
print("✅ Predictions saved")
plt.close()

# Save model
torch.save(cifar_model.state_dict(), 'D:/ai_engineering/week3_pytorch_basics/cifar10_model.pth')
print("\n✅ Model saved: cifar10_model.pth")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. CNNS FOR IMAGES:
   - Convolutional layers detect features
   - Pooling reduces spatial dimensions
   - Fully connected layers for classification

2. CONV LAYER:
   - Kernel/filter slides over image
   - Output size: (W - K + 2P) / S + 1
   - W=input width, K=kernel, P=padding, S=stride

3. POOLING:
   - MaxPool: Takes maximum value
   - AvgPool: Takes average
   - Reduces parameters, prevents overfitting

4. BATCH NORMALIZATION:
   - Normalizes layer inputs
   - Speeds up training
   - Acts as regularization

5. DATA AUGMENTATION:
   - RandomFlip, RandomCrop
   - Increases dataset size artificially
   - Improves generalization

6. CNN ARCHITECTURE PATTERN:
   - Conv → BatchNorm → ReLU → Pool (repeat)
   - Flatten
   - FC → Dropout → FC → Output

7. CIFAR-10 RESULTS:
   - 10 classes, 32x32 color images
   - ~60-75% accuracy with simple CNN
   - Can reach 90%+ with deeper networks

PERFORMANCE:
- Training time: ~2-3 min per epoch (CPU)
- Test accuracy: 65-75%
- Parameters: ~500K

NEXT: Transfer Learning with Pre-trained Models!
""")
print("="*70)

print("\n✅ Week 3 Complete! Ready for Week 4: Transfer Learning")