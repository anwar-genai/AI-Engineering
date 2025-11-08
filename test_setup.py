"""
Test Setup Script - Phase 1
Run this to verify all installations work correctly
Save as: test_setup.py
"""

import sys
print("Python version:", sys.version)
print("\n" + "="*50)
print("TESTING LIBRARY INSTALLATIONS")
print("="*50 + "\n")

# Test 1: NumPy and Pandas
try:
    import numpy as np
    import pandas as pd
    print("✅ NumPy version:", np.__version__)
    print("✅ Pandas version:", pd.__version__)
    
    # Quick test
    arr = np.array([1, 2, 3])
    df = pd.DataFrame({'A': [1, 2, 3]})
    print("   Test passed: NumPy and Pandas working\n")
except Exception as e:
    print("❌ NumPy/Pandas error:", e, "\n")

# Test 2: Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    print("✅ Matplotlib version:", matplotlib.__version__)
    print("   Test passed: Matplotlib working\n")
except Exception as e:
    print("❌ Matplotlib error:", e, "\n")

# Test 3: Scikit-learn
try:
    import sklearn
    print("✅ Scikit-learn version:", sklearn.__version__)
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(f"   Test passed: Loaded iris dataset ({iris.data.shape[0]} samples)\n")
except Exception as e:
    print("❌ Scikit-learn error:", e, "\n")

# Test 4: PyTorch
try:
    import torch
    print("✅ PyTorch version:", torch.__version__)
    print("   CUDA available:", torch.cuda.is_available())
    print("   (Should be False - we're using CPU version)")
    
    # Quick tensor test
    x = torch.randn(3, 3)
    print("   Test passed: Created tensor of shape", x.shape, "\n")
except Exception as e:
    print("❌ PyTorch error:", e, "\n")

# Test 5: PIL and OpenCV
try:
    from PIL import Image
    import cv2
    print("✅ PIL (Pillow) installed")
    print("✅ OpenCV version:", cv2.__version__)
    print("   Test passed: Image libraries working\n")
except Exception as e:
    print("❌ PIL/OpenCV error:", e, "\n")

# Test 6: pytesseract (Python wrapper)
try:
    import pytesseract
    print("✅ pytesseract wrapper installed")
    print("   (Tesseract engine must be installed separately!)\n")
except Exception as e:
    print("❌ pytesseract error:", e, "\n")

# Test 7: Check if Tesseract OCR engine is installed
try:
    import pytesseract
    from PIL import Image
    import numpy as np
    
    # Try to configure Tesseract path (common Windows location)
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    
    # Create a simple test image
    test_img = Image.new('RGB', (200, 50), color='white')
    
    # Try to run OCR (will fail if Tesseract not installed)
    version = pytesseract.get_tesseract_version()
    print("✅ Tesseract OCR engine found!")
    print(f"   Version: {version}")
    print("   Location: C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n")
except FileNotFoundError:
    print("⚠️  Tesseract OCR engine NOT found!")
    print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Install to: C:\\Program Files\\Tesseract-OCR")
    print("   Then re-run this test\n")
except Exception as e:
    print("⚠️  Tesseract test error:", e, "\n")

# Test 8: NLTK
try:
    import nltk
    print("✅ NLTK version:", nltk.__version__)
    print("   Test passed: NLP library working\n")
except Exception as e:
    print("❌ NLTK error:", e, "\n")

# Summary
print("="*50)
print("SETUP TEST COMPLETE")
print("="*50)
print("\n📋 NEXT STEPS:")
print("1. If any ❌ errors, reinstall that library")
print("2. If Tesseract ⚠️ warning, download and install Tesseract OCR")
print("3. If all ✅ green, you're ready to start Week 1!")
print("\nRun this anytime to verify your setup.")