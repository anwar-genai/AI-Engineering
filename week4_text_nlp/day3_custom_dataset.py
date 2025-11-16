"""
WEEK 4 - DAY 3-4: Working with Custom Datasets
==============================================
Learn to create custom datasets and train models on your own images



Topics:
- Creating custom Dataset class
- ImageFolder for organized data
- Data splitting (train/val/test)
- Advanced augmentation
- Training on custom data
- Inference on new images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil

print("="*70)
print("WEEK 4 - DAY 3-4: Custom Datasets")
print("="*70)

# Base directory for this script (week4_text_nlp)
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================
# PART 1: Create Synthetic Dataset
# ============================================
print("\n>>> PART 1: Creating Synthetic Image Dataset")

# Create directory structure under week4_text_nlp/datasets/custom_shapes
base_path = PROJECT_ROOT / "datasets" / "custom_shapes"
for split in ['train', 'val', 'test']:
    for category in ['circles', 'squares', 'triangles']:
        (base_path / split / category).mkdir(parents=True, exist_ok=True)

print(f"\nDataset structure created at: {base_path}")

def generate_shape_image(shape_type, size=64):
    """Generate simple geometric shapes"""
    img = Image.new('RGB', (size, size), color='white')
    pixels = img.load()
    center = size // 2
    radius = size // 3
    
    if shape_type == 'circle':
        for x in range(size):
            for y in range(size):
                if (x - center)**2 + (y - center)**2 <= radius**2:
                    pixels[x, y] = (255, 0, 0)  # Red circle
    
    elif shape_type == 'square':
        start = center - radius
        end = center + radius
        for x in range(start, end):
            for y in range(start, end):
                if 0 <= x < size and 0 <= y < size:
                    pixels[x, y] = (0, 255, 0)  # Green square
    
    elif shape_type == 'triangle':
        for x in range(size):
            for y in range(center, size):
                if abs(x - center) <= (y - center):
                    pixels[x, y] = (0, 0, 255)  # Blue triangle
    
    return img

# Generate dataset
print("\nGenerating synthetic images...")
shapes = ['circles', 'squares', 'triangles']
splits = {
    'train': 100,  # 100 images per class
    'val': 20,     # 20 images per class
    'test': 20     # 20 images per class
}

for split_name, num_images in splits.items():
    for shape in shapes:
        for i in range(num_images):
            img = generate_shape_image(shape[:-1])  # Remove 's' from plural
            img_path = base_path / split_name / shape / f'{shape}_{i:03d}.png'
            img.save(img_path)

print(f"Generated {sum(splits.values()) * 3} images")
print(f"   Train: {splits['train'] * 3}")
print(f"   Val: {splits['val'] * 3}")
print(f"   Test: {splits['test'] * 3}")

# Visualize samples
print("\nVisualizing generated samples...")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for idx, shape in enumerate(shapes):
    img_path = base_path / 'train' / shape / f'{shape}_000.png'
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(shape.capitalize())
    axes[idx].axis('off')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "day3_shape_samples.png", dpi=150)
print("Samples saved")
plt.close()

# ============================================
# PART 2: Custom Dataset Class
# ============================================
print("\n>>> PART 2: Creating Custom Dataset Class")

class CustomImageDataset(Dataset):
    """
    Custom Dataset class for loading images from folder structure
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Test custom dataset
test_dataset_obj = CustomImageDataset(base_path / 'train')
print(f"\nCustom Dataset Info:")
print(f"Number of samples: {len(test_dataset_obj)}")
print(f"Classes: {test_dataset_obj.classes}")
print(f"Class to index: {test_dataset_obj.class_to_idx}")

# ============================================
# PART 3: Data Transforms and Augmentation
# ============================================
print("\n>>> PART 3: Data Transforms and Augmentation")

# Training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

print("\nTraining transforms (with augmentation):")
print(train_transforms)
print("\nValidation transforms (no augmentation):")
print(val_transforms)

# Visualize augmentation
print("\nVisualizing data augmentation...")
sample_img, _ = test_dataset_obj[0]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(8):
    augmented = train_transforms(sample_img)
    # Denormalize
    augmented = augmented * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    augmented = augmented + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    augmented = torch.clamp(augmented, 0, 1)
    augmented = augmented.permute(1, 2, 0).numpy()
    
    axes[i].imshow(augmented)
    axes[i].set_title(f'Augmentation {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "day3_augmentation.png", dpi=150)
print("Augmentation examples saved")
plt.close()

# ============================================
# PART 4: Create DataLoaders
# ============================================
print("\n>>> PART 4: Creating DataLoaders")

# Create datasets
train_dataset = CustomImageDataset(base_path / 'train', transform=train_transforms)
val_dataset = CustomImageDataset(base_path / 'val', transform=val_transforms)
test_dataset = CustomImageDataset(base_path / 'test', transform=val_transforms)

print(f"\nDataset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Val: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ============================================
# PART 5: Build Model for Custom Dataset
# ============================================
print("\n>>> PART 5: Building Model")

class ShapeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ShapeClassifier, self).__init__()
        
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Create model
model = ShapeClassifier(num_classes=3)
print("\nModel created:")
print(f"Pre-trained: ResNet18")
print(f"Classes: {train_dataset.classes}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ============================================
# PART 6: Training with Validation
# ============================================
print("\n>>> PART 6: Training Model")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

# Training loop
num_epochs = 15
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | LR")
print("-" * 70)

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), PROJECT_ROOT / "best_shape_model.pth")
    
    print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | "
          f"{val_loss:8.4f} | {val_acc:7.2f}% | {current_lr:.6f}")

print(f"\nTraining complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# ============================================
# PART 7: Evaluation on Test Set
# ============================================
print("\n>>> PART 7: Test Set Evaluation")

# Load best model
model.load_state_dict(torch.load(PROJECT_ROOT / "best_shape_model.pth"))
test_loss, test_acc = validate(model, test_loader, criterion)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# Per-class accuracy
class_correct = [0] * 3
class_total = [0] * 3

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print("\nPer-class accuracy:")
for i, class_name in enumerate(train_dataset.classes):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{class_name:10s}: {acc:5.2f}%")

# ============================================
# PART 8: Visualization
# ============================================
print("\n>>> PART 8: Visualizing Results")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(train_accs, label='Train Acc', linewidth=2)
axes[1].plot(val_accs, label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "day3_training_curves.png", dpi=150)
print("Training curves saved")
plt.close()

# Visualize predictions
model.eval()
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

with torch.no_grad():
    # Collect up to 15 individual test images
    images_to_show = []
    for inputs, labels in test_loader:
        for img, label in zip(inputs, labels):
            images_to_show.append((img, label))
            if len(images_to_show) >= 15:
                break
        if len(images_to_show) >= 15:
            break

    # Plot predictions for collected images
    for ax, (img, label) in zip(axes.flat, images_to_show):
        output = model(img.unsqueeze(0))
        _, predicted = torch.max(output, 1)

        # Denormalize
        img_display = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img_display + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()

        ax.imshow(img_display)
        color = 'green' if predicted.item() == label.item() else 'red'
        ax.set_title(
            f'True: {train_dataset.classes[label]}\n'
            f'Pred: {train_dataset.classes[predicted.item()]}',
            color=color,
            fontsize=9,
        )
        ax.axis('off')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "day3_predictions.png", dpi=150)
print(" Predictions saved")
plt.close()

# ============================================
# PART 9: Inference on Single Image
# ============================================
print("\n>>> PART 9: Single Image Inference")

def predict_image(image_path, model, transform, classes):
    """Predict class for a single image"""
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'class': classes[predicted.item()],
        'confidence': confidence.item() * 100,
        'probabilities': {classes[i]: probabilities[0][i].item() * 100 
                         for i in range(len(classes))}
    }

# Test on random image
test_img_path = base_path / 'test' / 'circles' / 'circles_000.png'
result = predict_image(test_img_path, model, val_transforms, train_dataset.classes)

print(f"\nPrediction for: {test_img_path.name}")
print(f"Predicted class: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"\nAll probabilities:")
for class_name, prob in result['probabilities'].items():
    print(f"  {class_name:10s}: {prob:5.2f}%")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. CUSTOM DATASETS:
   - Organize: data/train/class1, data/train/class2, etc.
   - Use torch.utils.data.Dataset for custom loading
   - Implement __len__ and __getitem__ methods

2. DATA ORGANIZATION:
   - train/: Training data (70-80%)
   - val/: Validation for hyperparameter tuning (10-15%)
   - test/: Final evaluation (10-15%)

3. DATA AUGMENTATION:
   - RandomFlip, RandomRotation, ColorJitter
   - Only for training data
   - Helps prevent overfitting
   - Increases effective dataset size

4. TRAINING WORKFLOW:
   1. Split data (train/val/test)
   2. Create transforms (augmentation for train)
   3. Create datasets and dataloaders
   4. Build/load model
   5. Train with validation
   6. Save best model
   7. Evaluate on test set

5. VALIDATION:
   - Monitor validation loss to detect overfitting
   - Save model with best validation accuracy
   - Use learning rate scheduling

6. LEARNING RATE SCHEDULING:
   - StepLR: Reduce LR every N epochs
   - ReduceLROnPlateau: Reduce when val loss plateaus
   - CosineAnnealingLR: Cosine schedule

7. INFERENCE:
   - Load best model weights
   - Apply same transforms as validation
   - Use torch.no_grad() to save memory

RESULTS:
- Shape dataset: 300 train, 60 val, 60 test
- Model: ResNet18 (pre-trained)
- Final accuracy: ~99-100% (simple dataset)
- Training time: ~2-3 min

NEXT: Advanced techniques and Google Colab!
""")
print("="*70)

print("\n Day 3-4 Complete! Ready for Week 4 final project")