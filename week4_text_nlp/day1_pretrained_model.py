"""
WEEK 4 - DAY 1-2: Transfer Learning with Pre-trained Models
===========================================================
Learn to use pre-trained models from torchvision


Topics:
- Loading pre-trained models (ResNet, VGG, MobileNet)
- Feature extraction vs fine-tuning
- Freezing/unfreezing layers
- Transfer learning on CIFAR-10
- Model comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import copy

print("="*70)
print("WEEK 4 - DAY 1-2: Transfer Learning with Pre-trained Models")
print("="*70)

# ============================================
# PART 1: Loading Pre-trained Models
# ============================================
print("\n>>> PART 1: Exploring Pre-trained Models")

# Available models
print("\nPopular Pre-trained Models:")
print("- ResNet (18, 34, 50, 101, 152)")
print("- VGG (11, 13, 16, 19)")
print("- MobileNet V2, V3")
print("- EfficientNet")
print("- DenseNet")

# Load ResNet18
print("\n--- Loading ResNet18 ---")
resnet18 = models.resnet18(pretrained=True)
print(f"Model: {type(resnet18).__name__}")
print(f"\nArchitecture:")
print(resnet18)

# Count parameters
total_params = sum(p.numel() for p in resnet18.parameters())
trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Check final layer
print(f"\nOriginal final layer (for ImageNet, 1000 classes):")
print(resnet18.fc)

# ============================================
# PART 2: Modifying Pre-trained Models
# ============================================
print("\n>>> PART 2: Modifying Models for Custom Tasks")

print("\n--- Method 1: Replace Final Layer ---")

# Create a copy
model_method1 = models.resnet18(pretrained=True)

# Replace final layer for CIFAR-10 (10 classes)
num_features = model_method1.fc.in_features
model_method1.fc = nn.Linear(num_features, 10)

print(f"Original FC: in={num_features}, out=1000")
print(f"New FC: in={num_features}, out=10")
print(f"\nNew final layer:")
print(model_method1.fc)

print("\n--- Method 2: Feature Extraction (Freeze Layers) ---")

# Create another copy
model_method2 = models.resnet18(pretrained=True)

# Freeze all layers
for param in model_method2.parameters():
    param.requires_grad = False

# Replace and unfreeze final layer
model_method2.fc = nn.Linear(num_features, 10)

trainable = sum(p.numel() for p in model_method2.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model_method2.parameters() if not p.requires_grad)

print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters: {frozen:,}")
print(f"Percentage trainable: {100*trainable/(trainable+frozen):.2f}%")

print("\n--- Method 3: Fine-tuning (Unfreeze Some Layers) ---")

model_method3 = models.resnet18(pretrained=True)

# Freeze early layers, unfreeze later layers
for name, param in model_method3.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace final layer
model_method3.fc = nn.Linear(num_features, 10)

trainable = sum(p.numel() for p in model_method3.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model_method3.parameters() if not p.requires_grad)

print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters: {frozen:,}")
print(f"Percentage trainable: {100*trainable/(trainable+frozen):.2f}%")

# ============================================
# PART 3: Data Preparation for Transfer Learning
# ============================================
print("\n>>> PART 3: Data Preparation")

# ImageNet normalization (standard for pre-trained models)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

print("\nLoading CIFAR-10...")
train_dataset = datasets.CIFAR10(
    'D:/ai_engineering/datasets',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    'D:/ai_engineering/datasets',
    train=False,
    download=True,
    transform=test_transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================
# PART 4: Training Function
# ============================================
print("\n>>> PART 4: Creating Training Function")

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=10, model_name="Model"):
    """
    Train a model and track performance
    """
    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"\nTraining {model_name}...")
    print("Epoch | Train Loss | Train Acc | Test Acc | Time")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Testing phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        elapsed = time.time() - start_time
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {test_acc:8.2f}% | {elapsed:.1f}s")
    
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc
    }

# ============================================
# PART 5: Experiment 1 - Feature Extraction
# ============================================
print("\n>>> PART 5: Experiment 1 - Feature Extraction")
print("(Training only the final layer)")

# Setup model
model_fe = models.resnet18(pretrained=True)

# Freeze all layers
for param in model_fe.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model_fe.fc.in_features
model_fe.fc = nn.Linear(num_features, 10)

# Loss and optimizer (only final layer parameters)
criterion = nn.CrossEntropyLoss()
optimizer_fe = optim.Adam(model_fe.fc.parameters(), lr=0.001)

# Train
results_fe = train_model(
    model_fe, train_loader, test_loader, 
    criterion, optimizer_fe, 
    num_epochs=5,
    model_name="Feature Extraction"
)

# Save model
torch.save(model_fe.state_dict(), 
           'D:/ai_engineering/week4_text_nlp/week4_transfer_learning/resnet18_feature_extraction.pth')
print("✅ Model saved: resnet18_feature_extraction.pth")

# ============================================
# PART 6: Experiment 2 - Fine-tuning
# ============================================
print("\n>>> PART 6: Experiment 2 - Fine-tuning")
print("(Training all layers with small learning rate)")

# Setup model
model_ft = models.resnet18(pretrained=True)

# Replace final layer
model_ft.fc = nn.Linear(num_features, 10)

# Optimizer (all parameters, small LR for pre-trained layers)
optimizer_ft = optim.Adam([
    {'params': model_ft.fc.parameters(), 'lr': 0.001},  # New layer: higher LR
    {'params': model_ft.layer4.parameters(), 'lr': 0.0001},  # Last conv: small LR
    {'params': model_ft.layer3.parameters(), 'lr': 0.00001},  # Earlier: tiny LR
], lr=0.00001)  # Default for rest

# Train
results_ft = train_model(
    model_ft, train_loader, test_loader,
    criterion, optimizer_ft,
    num_epochs=5,
    model_name="Fine-tuning"
)

# Save model
torch.save(model_ft.state_dict(),
           'D:/ai_engineering/week4_text_nlp/week4_transfer_learning/resnet18_finetuned.pth')
print("✅ Model saved: resnet18_finetuned.pth")

# ============================================
# PART 7: Experiment 3 - Training from Scratch
# ============================================
print("\n>>> PART 7: Experiment 3 - Training from Scratch")
print("(For comparison - no pre-trained weights)")

# Setup model (pretrained=False)
model_scratch = models.resnet18(pretrained=False)
model_scratch.fc = nn.Linear(num_features, 10)

# Optimizer (all parameters)
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)

# Train
results_scratch = train_model(
    model_scratch, train_loader, test_loader,
    criterion, optimizer_scratch,
    num_epochs=5,
    model_name="From Scratch"
)

# ============================================
# PART 8: Compare Results
# ============================================
print("\n>>> PART 8: Comparing All Methods")

comparison = {
    'Feature Extraction': results_fe,
    'Fine-tuning': results_ft,
    'From Scratch': results_scratch
}

# Print summary
print("\nFinal Test Accuracies:")
print("-" * 40)
for name, results in comparison.items():
    print(f"{name:20s}: {results['best_acc']:.2f}%")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Loss
ax1 = axes[0]
for name, results in comparison.items():
    ax1.plot(results['train_losses'], label=name, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training Accuracy
ax2 = axes[1]
for name, results in comparison.items():
    ax2.plot(results['train_accs'], label=name, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Test Accuracy
ax3 = axes[2]
for name, results in comparison.items():
    ax3.plot(results['test_accs'], label=name, linewidth=2, marker='o')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Test Accuracy Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week4_text_nlp/week4_transfer_learning/day1_comparison.png', dpi=150)
print("\n Comparison plot saved")
plt.close()

# ============================================
# PART 9: Visualize Predictions
# ============================================
print("\n>>> PART 9: Visualizing Predictions")

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Use best model (fine-tuned)
model_ft.eval()

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        image, label = test_dataset[i * 100]
        output = model_ft(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        
        # Denormalize for display
        img = image.clone()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img)
        color = 'green' if predicted.item() == label else 'red'
        ax.set_title(f'True: {classes[label]}\nPred: {classes[predicted.item()]}',
                    color=color, fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('D:/ai_engineering/week4_text_nlp/week4_transfer_learning/day1_predictions.png', dpi=150)
print(" Predictions saved")
plt.close()

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. TRANSFER LEARNING:
   - Use pre-trained models as starting point
   - Much faster than training from scratch
   - Works with smaller datasets

2. THREE APPROACHES:
   
   a) Feature Extraction:
      - Freeze all layers except final
      - Fastest training
      - Good for small datasets
      - Accuracy: ~75-80%
   
   b) Fine-tuning:
      - Train all layers with small LR
      - Best performance
      - Needs more time
      - Accuracy: ~80-85%
   
   c) From Scratch:
      - No pre-trained weights
      - Needs lots of data
      - Slowest to converge
      - Accuracy: ~60-70% (with limited epochs)

3. WHEN TO USE WHAT:
   - Small dataset (<10k images): Feature Extraction
   - Medium dataset (10k-100k): Fine-tuning
   - Large dataset (>100k): Consider training from scratch

4. LEARNING RATES:
   - Pre-trained layers: 10-100x smaller LR
   - New layers: Normal LR
   - Prevents forgetting learned features

5. DATA PREPROCESSING:
   - Use ImageNet normalization for pre-trained models
   - mean=[0.485, 0.456, 0.406]
   - std=[0.229, 0.224, 0.225]

RESULTS ON CIFAR-10 (5 epochs):
- Feature Extraction: ~77%
- Fine-tuning: ~82%
- From Scratch: ~65%

NEXT: Custom datasets and advanced techniques!
""")
print("="*70)

print("\n Day 1-2 Complete! Move to day3_custom_dataset.py")