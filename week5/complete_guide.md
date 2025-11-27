# 🚀 Week 5-6: Google Colab & GPU Training - Complete Guide

**Duration:** 1-2 weeks  
**Goal:** Master GPU training with Google Colab, achieve 90%+ on CIFAR-10  
**Hardware:** Google Colab (Free GPU: Tesla T4, 15GB VRAM)

---

## 📋 Table of Contents

1. [Getting Started with Google Colab](#getting-started)
2. [Day 1: Colab Setup](#day-1)
3. [Day 2-3: Training Large Models](#day-2-3)
4. [Day 4-5: Model Optimization](#day-4-5)
5. [Day 6-7: Deployment Basics](#day-6-7)
6. [Tips & Troubleshooting](#tips)

---

## 🎯 Getting Started with Google Colab {#getting-started}

### **What is Google Colab?**
- Free cloud computing platform by Google
- Access to FREE GPU (Tesla T4, ~15GB VRAM)
- Access to FREE TPU (optional, even faster)
- Pre-installed ML libraries (PyTorch, TensorFlow)
- Jupyter notebook interface
- 12-hour session limit

### **Why Use Colab?**
- ✅ Your laptop (CPU only) is 10-100x slower
- ✅ No GPU purchase needed ($500-2000 saved!)
- ✅ Train larger models (ResNet50, EfficientNet)
- ✅ Essential for OCR models (Phase 3)
- ✅ Industry-standard workflow

### **Colab vs Your Laptop**

| Task | Your Laptop (CPU) | Google Colab (GPU) | Speedup |
|------|-------------------|-------------------|---------|
| ResNet18 training | 5 min/epoch | 30 sec/epoch | 10x |
| ResNet50 training | 20 min/epoch | 1.5 min/epoch | 13x |
| Large dataset (100K images) | Hours | Minutes | 50-100x |

---

## 📅 Day 1: Google Colab Setup {#day-1}

### **Step 1: Access Google Colab**

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with Google account
3. Click **"New Notebook"**

**You should see:**
```
A Jupyter notebook interface
Cell with "Code" dropdown
Menu bar: File, Edit, View, Insert, Runtime, Tools, Help
```

---

### **Step 2: Enable GPU**

**CRITICAL STEP:**

1. Click **Runtime** → **Change runtime type**
2. Hardware accelerator: Select **GPU**
3. Click **Save**
4. You'll see: "Runtime changed to Python 3 with GPU"

**Verify GPU:**
```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should print: Tesla T4 (or similar)
```

---

### **Step 3: Mount Google Drive**

**Why?** To save your models and not lose them after 12 hours

```python
from google.colab import drive
drive.mount('/content/drive')
```

**You'll see:**
- A popup asking for permission
- Click the link
- Choose your Google account
- Copy the authorization code
- Paste in Colab

**Verify:**
```python
import os
os.listdir('/content/drive/MyDrive')  # Shows your Drive files
```

---

### **Step 4: Create Project Structure**

```python
import os

# Create project directory in Drive
project_dir = '/content/drive/MyDrive/ai_engineering/week5_colab'
os.makedirs(project_dir, exist_ok=True)

# Create subdirectories
os.makedirs(f'{project_dir}/models', exist_ok=True)
os.makedirs(f'{project_dir}/results', exist_ok=True)
os.makedirs(f'{project_dir}/checkpoints', exist_ok=True)

print("✅ Project structure created!")
print(f"Location: {project_dir}")
```

**Your Drive structure:**
```
MyDrive/
└── ai_engineering/
    └── week5_colab/
        ├── models/          # Saved models
        ├── results/         # Plots, logs
        └── checkpoints/     # Training checkpoints
```

---

### **Step 5: Test GPU Speed**

Copy the entire "Day 1" script I provided above. It includes:
- GPU vs CPU speed comparison
- Memory management
- Basic training example
- Helper functions

**Expected output:**
```
GPU is 15-30x faster for matrix operations!
```

---

## 🏋️ Day 2-3: Training Large Models {#day-2-3}

### **What You'll Achieve:**
- Train ResNet50 (25M parameters)
- Use advanced data augmentation
- Achieve 90%+ accuracy on CIFAR-10
- Save models to Google Drive

### **Step-by-Step:**

**1. Create New Notebook**
- File → New Notebook
- Name it: "Week5_ResNet50_Training"
- Enable GPU (Runtime → Change runtime type → GPU)

**2. Copy the "Day 2-3" Script**
- Copy the entire script I provided
- Paste into cells (one section per cell)
- Run cells sequentially (Shift+Enter)

**3. Start Training**
```python
# The script will automatically:
# ✅ Download CIFAR-10
# ✅ Setup data augmentation
# ✅ Build ResNet50
# ✅ Train for 30 epochs (~45-60 minutes)
# ✅ Save best model to Drive
```

**Expected Timeline:**
- Setup: 2-3 minutes
- Training: 45-60 minutes (30 epochs × ~2 min/epoch)
- Evaluation: 2 minutes
- **Total: ~1 hour**

**Expected Accuracy:**
- After 10 epochs: ~85%
- After 20 epochs: ~88%
- After 30 epochs: ~90-92%

---

### **While Training: What to Monitor**

**Good Signs:**
```
✅ Loss decreasing smoothly
✅ Val accuracy increasing
✅ Train and val accuracy close (within 5%)
✅ GPU memory: 4-6 GB / 15 GB used
```

**Warning Signs:**
```
⚠️ Loss not decreasing → Check learning rate
⚠️ Val acc much lower than train → Overfitting
⚠️ GPU memory 14/15 GB → Reduce batch size
⚠️ "CUDA out of memory" → Restart, reduce batch size
```

---

## ⚡ Day 4-5: Model Optimization {#day-4-5}

### **Topics Covered:**
- Mixed precision training (2x faster)
- Gradient accumulation (larger effective batch size)
- Model pruning (smaller models)
- Knowledge distillation

### **Mixed Precision Training**

**What:** Use FP16 instead of FP32 (half precision)  
**Benefit:** 2x faster, uses less memory  
**Drawback:** Tiny accuracy loss (0.1-0.5%)

```python
from torch.cuda.amp import autocast, GradScaler

# Create scaler
scaler = GradScaler()

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Forward pass with autocast
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward pass with scaler
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Results:**
- Training time: 1.5 min/epoch → **0.8 min/epoch** (2x faster!)
- Accuracy: 90.5% → 90.2% (tiny drop)

---

### **Gradient Accumulation**

**Problem:** Batch size limited by GPU memory  
**Solution:** Accumulate gradients over multiple batches

```python
accumulation_steps = 4  # Effective batch size: 128 × 4 = 512

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # Normalize
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit:** Can use larger effective batch size → better convergence

---

## 🚀 Day 6-7: Deployment Basics {#day-6-7}

### **Download Model from Colab**

```python
from google.colab import files

# Download best model
files.download('/content/drive/MyDrive/ai_engineering/week5_colab/models/best_resnet50.pth')
```

---

### **Inference Script (Run Locally)**

Save this as `inference.py` on your laptop:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model
def load_model(checkpoint_path, device='cpu'):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model

# Predict single image
def predict(image_path, model, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return classes[predicted.item()], confidence.item()

# Usage
if __name__ == '__main__':
    model = load_model('best_resnet50.pth', device='cpu')
    
    class_name, confidence = predict('test_image.jpg', model)
    print(f"Prediction: {class_name} ({confidence*100:.2f}% confidence)")
```

**Run locally:**
```bash
python inference.py
```

---

## 💡 Tips & Troubleshooting {#tips}

### **Colab Session Management**

**Problem:** Session disconnects after 12 hours  
**Solution:**
1. Save checkpoints every 5 epochs
2. Use this code to prevent disconnection:
```python
# Run this in a cell (keeps session alive)
import time
from IPython.display import Javascript

def keep_alive():
    while True:
        display(Javascript('navigator.webdriver'))
        time.sleep(60)

# Start in background
import threading
t = threading.Thread(target=keep_alive)
t.start()
```

---

### **GPU Memory Issues**

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# 1. Reduce batch size
train_loader = DataLoader(dataset, batch_size=64)  # Instead of 128

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Use gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Reduce model size
model = models.resnet18(pretrained=True)  # Instead of resnet50
```

---

### **Slow Data Loading**

**Problem:** Training slow, GPU underutilized

**Solution:**
```python
# Increase num_workers
train_loader = DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=4,  # Increase this (try 2, 4, 8)
    pin_memory=True  # Faster GPU transfer
)
```

---

### **Saving/Loading Models**

**Save:**
```python
# Save complete checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'train_losses': train_losses,
    'val_losses': val_losses
}, '/content/drive/MyDrive/checkpoint.pth')
```

**Load:**
```python
checkpoint = torch.load('/content/drive/MyDrive/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

### **Resume Training**

```python
# Check if checkpoint exists
checkpoint_path = '/content/drive/MyDrive/checkpoint.pth'

if os.path.exists(checkpoint_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("Starting from scratch...")
    start_epoch = 0

# Train
for epoch in range(start_epoch, num_epochs):
    # Training code...
```

---

## 📊 Expected Results Summary

### **CIFAR-10 Accuracy Progression**

| Week | Method | Model | Accuracy | Training Time |
|------|--------|-------|----------|---------------|
| 3 | From scratch (CPU) | Custom CNN | 65-70% | 30 min |
| 3 | From scratch (CPU) | ResNet18 | 70-75% | 2 hours |
| 4 | Transfer learning (CPU) | ResNet18 | 82% | 30 min |
| 5 | Transfer learning (GPU) | ResNet50 | 90-92% | 1 hour |
| 5 | Mixed precision (GPU) | ResNet50 | 90% | 30 min |

**Key Insight:** GPU + Larger Model = Best accuracy in less time!

---

## ✅ Week 5-6 Checklist

### **Day 1: Setup**
- [ ] Access Google Colab
- [ ] Enable GPU
- [ ] Mount Google Drive
- [ ] Create project structure
- [ ] Run GPU speed test
- [ ] Understand device management

### **Day 2-3: Training**
- [ ] Load CIFAR-10 with advanced augmentation
- [ ] Build ResNet50
- [ ] Train for 30 epochs
- [ ] Achieve 90%+ accuracy
- [ ] Save model to Drive
- [ ] Visualize results

### **Day 4-5: Optimization**
- [ ] Implement mixed precision training
- [ ] Try gradient accumulation
- [ ] Compare training times
- [ ] Experiment with different models

### **Day 6-7: Deployment**
- [ ] Download trained model
- [ ] Create inference script
- [ ] Test on local machine
- [ ] Predict new images
- [ ] Document results

---

## 🎯 Success Criteria

**You've mastered Week 5-6 when you can:**
- ✅ Set up and use Google Colab GPU
- ✅ Train large models (ResNet50) efficiently
- ✅ Achieve 90%+ accuracy on CIFAR-10
- ✅ Save/load models from Google Drive
- ✅ Resume training from checkpoints
- ✅ Deploy models locally for inference
- ✅ Handle GPU memory efficiently

---

## ➡️ Next Steps: Week 7-8

**Advanced Topics:**
- Learning rate finding
- Ensemble methods
- Test time augmentation
- Model compression
- ONNX export
- Production deployment

**Then:** Phase 3 - OCR & Multimodal AI! 🎯

---

## 📚 Resources

### **Google Colab**
- [Official Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab Pro](https://colab.research.google.com/signup) ($9.99/month - faster GPUs, longer sessions)

### **PyTorch on GPU**
- [PyTorch CUDA Guide](https://pytorch.org/docs/stable/cuda.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

### **Model Zoo**
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Pretrained Models](https://pytorch.org/serve/model_zoo.html)

---

**Week 5-6 Status:** Ready to start! 🚀  
**Estimated Time:** 1-2 weeks  
**Difficulty:** Intermediate  
**Reward:** 90%+ accuracy + GPU skills = Ready for OCR! 🎉