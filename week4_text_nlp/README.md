# 🎯 Week 4: Transfer Learning & Fine-tuning

**Duration:** ~15-20 hours  
**Status:** ✅ Complete  
**Goal:** Master transfer learning, work with pre-trained models, and build custom image classifiers

---

## 📚 Learning Objectives

By the end of Week 4, you should be able to:
- ✅ Load and use pre-trained models (ResNet, VGG, MobileNet)
- ✅ Understand feature extraction vs fine-tuning
- ✅ Freeze/unfreeze layers strategically
- ✅ Create custom datasets from your own images
- ✅ Implement advanced data augmentation
- ✅ Use learning rate scheduling
- ✅ Achieve 80%+ accuracy on CIFAR-10
- ✅ Build production-ready image classifiers

---

## 📁 Files in This Directory

```
week4_transfer_learning/
├── README.md                              # This file
├── day1_pretrained_models.py              # Transfer learning basics
├── day3_custom_dataset.py                 # Custom datasets
├── day1_comparison.png                    # Method comparison
├── day1_predictions.png                   # Model predictions
├── day3_shape_samples.png                 # Custom dataset samples
├── day3_augmentation.png                  # Augmentation examples
├── day3_training_curves.png               # Training visualization
├── day3_predictions.png                   # Custom model predictions
├── resnet18_feature_extraction.pth        # Saved model (feature extraction)
├── resnet18_finetuned.pth                 # Saved model (fine-tuned)
└── best_shape_model.pth                   # Best custom model
```

---

## 🎯 Day-by-Day Breakdown

### **Day 1-2: Transfer Learning Fundamentals**
**File:** `day1_pretrained_models.py`

**What You'll Learn:**
- Loading pre-trained models from torchvision
- Three transfer learning approaches
- Freezing and unfreezing layers
- Learning rate strategies
- Comparing transfer learning methods

**Three Transfer Learning Approaches:**

**1. Feature Extraction (Freeze All Layers)**
```python
# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (only this will train)
model.fc = nn.Linear(512, num_classes)

# Train only final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```
- **When:** Small dataset (<10k images), similar to ImageNet
- **Pros:** Fast training, prevents overfitting
- **Cons:** Limited adaptation to new domain
- **CIFAR-10 Result:** ~77% accuracy

**2. Fine-tuning (Train All Layers)**
```python
# Load pre-trained model
model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(512, num_classes)

# Train all layers with different learning rates
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.layer3.parameters(), 'lr': 0.00001}
], lr=0.00001)
```
- **When:** Medium dataset (10k-100k images)
- **Pros:** Best performance, adapts to new domain
- **Cons:** Slower training, risk of overfitting
- **CIFAR-10 Result:** ~82% accuracy

**3. Training from Scratch**
```python
# No pre-trained weights
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, num_classes)

# Train everything
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- **When:** Large dataset (>100k images), very different domain
- **Pros:** Full customization
- **Cons:** Needs lots of data, slow convergence
- **CIFAR-10 Result:** ~65% accuracy (5 epochs)

**Key Insights:**
- Transfer learning gives 10-20% accuracy boost
- Pre-trained features work across domains
- Fine-tuning > Feature Extraction > From Scratch (usually)

---

### **Day 3-4: Custom Datasets**
**File:** `day3_custom_dataset.py`

**What You'll Learn:**
- Creating custom Dataset class
- Organizing your own images
- Advanced data augmentation
- Train/val/test splitting
- Building production classifiers

**Dataset Organization:**
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── class3/
│       └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── class3/
└── test/
    ├── class1/
    ├── class2/
    └── class3/
```

**Custom Dataset Class:**
```python
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths
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
```

**Advanced Data Augmentation:**
```python
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Learning Rate Scheduling:**
```python
# Step LR: Reduce every N epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Reduce on Plateau: Reduce when metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3
)

# Cosine Annealing: Smooth reduction
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=0.00001
)
```

**Project: Shape Classifier**
- Created synthetic dataset: circles, squares, triangles
- 300 training images, 60 validation, 60 test
- ResNet18 with transfer learning
- Result: 99-100% accuracy (simple dataset)

---

## 📊 Skills Demonstrated

### **Transfer Learning**
- ✅ Loading pre-trained models
- ✅ Feature extraction approach
- ✅ Fine-tuning approach
- ✅ Layer freezing/unfreezing
- ✅ Differential learning rates

### **Custom Datasets**
- ✅ Custom Dataset class
- ✅ Data organization
- ✅ Train/val/test splitting
- ✅ ImageFolder pattern

### **Data Augmentation**
- ✅ RandomFlip, RandomRotation
- ✅ ColorJitter
- ✅ RandomAffine
- ✅ RandomPerspective
- ✅ Normalization

### **Training Techniques**
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Model checkpointing
- ✅ Validation monitoring

### **Inference**
- ✅ Single image prediction
- ✅ Confidence scores
- ✅ Class probabilities
- ✅ Batch prediction

---

## 🚀 How to Run

### **Prerequisites**
```bash
pip install torch torchvision pillow matplotlib
```

### **Execute Scripts**
```bash
# Activate environment
cd D:\ai_engineering
ai_env\Scripts\activate

# Day 1-2: Transfer learning comparison
python week4_transfer_learning/day1_pretrained_models.py

# Day 3-4: Custom dataset
python week4_transfer_learning/day3_custom_dataset.py
```

### **Expected Runtime**
- Day 1-2: ~15-25 minutes (trains 3 models)
- Day 3-4: ~10-15 minutes (synthetic dataset + training)

---

## 📈 Performance Comparison

### **CIFAR-10 Results (5 epochs)**

| Method | Train Acc | Test Acc | Training Time | Parameters Trained |
|--------|-----------|----------|---------------|-------------------|
| Feature Extraction | 85% | 77% | 5 min | 5,130 (0.05%) |
| Fine-tuning | 88% | 82% | 15 min | 11,181,642 (100%) |
| From Scratch | 70% | 65% | 15 min | 11,181,642 (100%) |

**Key Observations:**
- Transfer learning gives +12-17% accuracy boost
- Feature extraction is 3x faster
- Fine-tuning achieves best accuracy
- From scratch needs more epochs to converge

---

## 🎓 Key Takeaways

### **When to Use Transfer Learning?**

**Always try transfer learning first!** It works when:
- ✅ Limited training data (<100k images)
- ✅ Similar visual concepts to ImageNet
- ✅ Want faster training
- ✅ Need good baseline quickly

**Train from scratch when:**
- ❌ Very different domain (medical, satellite, microscopy)
- ❌ Have massive dataset (>1M images)
- ❌ Need specialized architecture

### **Feature Extraction vs Fine-tuning**

**Use Feature Extraction when:**
- Small dataset (<10k images)
- Data very similar to ImageNet
- Need fast experimentation
- Limited compute resources

**Use Fine-tuning when:**
- Medium dataset (10k-100k images)
- Data somewhat different from ImageNet
- Want maximum accuracy
- Have compute budget

### **Learning Rate Strategy**

**For Transfer Learning:**
```python
# New layers: Normal LR
# Layer 4 (closest to output): Small LR (0.1x)
# Layer 3: Tiny LR (0.01x)
# Layer 2: Tiny LR (0.01x)
# Layer 1 (closest to input): Tiny LR (0.01x) or frozen

optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.layer3.parameters(), 'lr': 0.00001},
], lr=0.00001)  # Default for remaining layers
```

**Why?**
- Early layers learn general features (edges, colors)
- Later layers learn task-specific features
- Don't want to destroy pre-trained early features

### **Data Augmentation Best Practices**

**Always use:**
- RandomHorizontalFlip (if applicable)
- RandomCrop with padding
- Normalization (ImageNet stats)

**Use carefully:**
- RandomRotation (not for text/numbers)
- ColorJitter (not for medical images)
- RandomPerspective (for documents/scenes)

**Never use on test set:**
- Only augment training data
- Val/test should match production

### **Validation Strategy**

**Train/Val/Test Split:**
- 70-80% Training
- 10-15% Validation (for hyperparameters)
- 10-15% Test (final evaluation only)

**Why separate val and test?**
- Val: Tune hyperparameters
- Test: Unbiased performance estimate
- Don't tune on test set!

---

## 🔄 Practice Exercises

### **Beginner**
1. Try different pre-trained models (VGG16, MobileNetV2)
2. Experiment with different learning rates
3. Add/remove data augmentation
4. Visualize layer activations

### **Intermediate**
1. Build custom dataset with your own photos
2. Implement progressive unfreezing
3. Compare different schedulers
4. Add mixup augmentation

### **Advanced**
1. Implement attention mechanisms
2. Use multiple models (ensemble)
3. Try knowledge distillation
4. Build multi-task learning

---

## 💡 Real-World Applications

### **1. Medical Image Classification**
```python
# Chest X-ray classification (COVID vs Normal)
model = models.resnet50(pretrained=True)

# Freeze early layers (general features)
for param in list(model.parameters())[:-30]:
    param.requires_grad = False

# Fine-tune on medical data
```

### **2. Product Classification (E-commerce)**
```python
# Classify product images
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    # Add watermark removal, background subtraction
])
```

### **3. Wildlife Monitoring**
```python
# Animal species classification from camera traps
# Challenge: Imbalanced classes, varying lighting
model = models.efficientnet_b0(pretrained=True)

# Use weighted loss for imbalanced data
weights = torch.tensor([0.1, 0.3, 0.6, ...])
criterion = nn.CrossEntropyLoss(weight=weights)
```

### **4. Document Classification**
```python
# Invoice vs Receipt vs Form
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomPerspective(0.2),  # Document perspective
    transforms.RandomRotation(5),  # Slight rotation
    transforms.Grayscale(num_output_channels=3),  # Often grayscale
])
```

---

## 🐛 Common Issues & Solutions

### **Issue 1: Model Overfitting**
**Symptoms:**
- Training accuracy >> Validation accuracy
- Validation loss increasing

**Solutions:**
```python
# 1. Add more dropout
self.dropout = nn.Dropout(0.5)

# 2. Increase data augmentation
transforms.RandomHorizontalFlip(p=0.8)

# 3. Use weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Early stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

### **Issue 2: Low Training Accuracy**
**Symptoms:**
- Both train and val accuracy low
- Loss not decreasing

**Solutions:**
```python
# 1. Increase learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 2. Reduce regularization
self.dropout = nn.Dropout(0.2)  # Instead of 0.5

# 3. Train longer
num_epochs = 50  # Instead of 10

# 4. Check data preprocessing
print(images.min(), images.max())  # Should be normalized
```

### **Issue 3: Poor Transfer from ImageNet**
**Symptoms:**
- Transfer learning worse than scratch
- Very different domain

**Solutions:**
```python
# 1. Use domain-specific pre-trained model
model = torch.hub.load('repo', 'medical_resnet50')

# 2. Fine-tune more layers
for param in model.parameters():
    param.requires_grad = True

# 3. Use smaller pre-trained model
model = models.resnet18(pretrained=True)  # Instead of resnet50
```

### **Issue 4: Slow Training**
```python
# Solutions:
# 1. Use smaller input size
transforms.Resize((128, 128))  # Instead of 224

# 2. Reduce batch size
DataLoader(dataset, batch_size=32)  # Instead of 128

# 3. Use mixed precision (if GPU)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 4. Fewer data augmentations
# Remove expensive operations like RandomPerspective
```

---

## 📚 Pre-trained Models Guide

### **Popular Models (torchvision.models)**

| Model | Parameters | ImageNet Top-1 | Speed | Use Case |
|-------|------------|----------------|-------|----------|
| **ResNet18** | 11M | 69.8% | Fast | General purpose, good baseline |
| **ResNet50** | 25M | 76.1% | Medium | Better accuracy, still fast |
| **MobileNetV2** | 3.5M | 71.9% | Very Fast | Mobile/embedded, real-time |
| **EfficientNet-B0** | 5M | 77.3% | Fast | Best accuracy/speed tradeoff |
| **VGG16** | 138M | 71.6% | Slow | Simple architecture, large |
| **DenseNet121** | 8M | 74.4% | Medium | Feature reuse, memory efficient |

**How to choose?**
- **Starting out:** ResNet18
- **Best performance:** EfficientNet or ResNet50
- **Mobile deployment:** MobileNetV2
- **Limited memory:** MobileNetV2 or EfficientNet-B0

---

## 🔗 Resources

### **Official Documentation**
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

### **Papers**
- [ImageNet Classification (ResNet)](https://arxiv.org/abs/1512.03385)
- [MobileNets](https://arxiv.org/abs/1704.04861)
- [EfficientNet](https://arxiv.org/abs/1905.11946)

### **Datasets**
- [ImageNet](https://www.image-net.org/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers With Code](https://paperswithcode.com/datasets)

---

## ✅ Completion Checklist

- [x] Understand transfer learning concepts
- [x] Load pre-trained models
- [x] Implement feature extraction
- [x] Implement fine-tuning
- [x] Compare all three approaches
- [x] Create custom Dataset class
- [x] Organize custom image data
- [x] Implement data augmentation
- [x] Use learning rate scheduling
- [x] Save and load models
- [ ] Apply to your own dataset
- [ ] Deploy model for inference

---

## ➡️ Next Steps: Week 5-6

**Google Colab & GPU Training:**
- Setting up Google Colab
- Using free GPU resources
- Training larger models
- Longer training runs
- Model deployment basics
- Creating inference APIs

**Advanced Topics:**
- Object detection
- Image segmentation
- Multi-task learning
- Few-shot learning
- Self-supervised learning

---

**Week 4 Status:** ✅ Complete  
**Best CIFAR-10 Accuracy:** 82% (Fine-tuning)  
**Custom Dataset Accuracy:** 99-100%  
**Time Invested:** ~15-20 hours  
**Ready for Advanced Topics:** Absolutely! 🚀  

---

**Last Updated:** November 2024  
**Next Challenge:** Google Colab & Production Deployment 