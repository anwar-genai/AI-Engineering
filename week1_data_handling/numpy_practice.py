"""
Week 1: NumPy Fundamentals
Save as: D:\ai_engineering\week1_data_handling\numpy_practice.py

Complete each exercise. Run the code to verify outputs.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("WEEK 1 - DAY 1-2: NumPy Exercises")
print("="*60)

# ============================================
# EXERCISE 1: Array Creation and Indexing
# ============================================
print("\n>>> Exercise 1: Array Basics")

# Task 1.1: Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.arange(0, 10, 2)  # 0 to 10, step 2
arr3 = np.linspace(0, 1, 5)  # 5 numbers between 0 and 1
arr4 = np.zeros((3, 3))
arr5 = np.ones((2, 4))
arr6 = np.eye(3)  # Identity matrix

print("Array 1:", arr1)
print("Array 2 (arange):", arr2)
print("Array 3 (linspace):", arr3)
print("Zeros:\n", arr4)
print("Ones:\n", arr5)
print("Identity:\n", arr6)

# Task 1.2: Array properties
print(f"\nArray 1 shape: {arr1.shape}")
print(f"Array 1 dtype: {arr1.dtype}")
print(f"Array 1 size: {arr1.size}")

# TODO: Create your own arrays
# Create a 4x4 array of random integers between 1 and 100
random_arr = np.random.randint(1, 101, size=(4, 4))
print("\nYour random array:\n", random_arr)

# ============================================
# EXERCISE 2: Array Operations
# ============================================
print("\n>>> Exercise 2: Array Operations")

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise operations
print("a + b =", a + b)
print("a * b =", a * b)
print("a ** 2 =", a ** 2)
print("sqrt(a) =", np.sqrt(a))

# Aggregation functions
data = np.array([5, 2, 9, 1, 7, 4])
print(f"\nData: {data}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Std: {np.std(data)}")
print(f"Max: {np.max(data)}")
print(f"Min: {np.min(data)}")

# TODO: Calculate statistics on your random array
print("\nStatistics on your random array:")
print(f"Mean: {np.mean(random_arr):.2f}")
print(f"Max: {np.max(random_arr)}")
print(f"Min: {np.min(random_arr)}")

# ============================================
# EXERCISE 3: Indexing and Slicing
# ============================================
print("\n>>> Exercise 3: Indexing & Slicing")

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
print("Original array:", arr)

# Basic indexing
print("First element:", arr[0])
print("Last element:", arr[-1])

# Slicing
print("First 3 elements:", arr[:3])
print("Last 3 elements:", arr[-3:])
print("Every other element:", arr[::2])

# 2D indexing
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("\nMatrix:\n", matrix)
print("Element at (1, 2):", matrix[1, 2])  # Row 1, Col 2
print("First row:", matrix[0, :])
print("Second column:", matrix[:, 1])

# Boolean indexing
print("\nBoolean indexing:")
data = np.array([5, 2, 9, 1, 7, 4])
print("Data:", data)
print("Elements > 4:", data[data > 4])
print("Even numbers:", data[data % 2 == 0])

# TODO: From your random array, extract:
# - All elements greater than 50
# - All elements in the last row
print("\nFrom your random array:")
print("Elements > 50:", random_arr[random_arr > 50])
print("Last row:", random_arr[-1, :])

# ============================================
# EXERCISE 4: Reshaping and Combining
# ============================================
print("\n>>> Exercise 4: Reshaping")

arr = np.arange(12)
print("Original (12,):", arr)

reshaped = arr.reshape(3, 4)
print("Reshaped to (3, 4):\n", reshaped)

reshaped2 = arr.reshape(2, 6)
print("Reshaped to (2, 6):\n", reshaped2)

# Flattening
print("Flattened back:", reshaped.flatten())

# Combining arrays
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("\nArray a:\n", a)
print("Array b:\n", b)

# Stack vertically
print("Vertical stack:\n", np.vstack([a, b]))

# Stack horizontally
print("Horizontal stack:\n", np.hstack([a, b]))

# ============================================
# MINI PROJECT: Image as NumPy Array
# ============================================
print("\n>>> Mini Project: Creating and Visualizing Images")

# Create a simple 100x100 grayscale image
img_size = 100
image = np.zeros((img_size, img_size))

# Draw a white square in the center
center = img_size // 2
square_size = 30
image[center-square_size:center+square_size, 
      center-square_size:center+square_size] = 255

print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Pixel values range: {image.min()} to {image.max()}")

# Visualize
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.title('NumPy Image: White Square')
plt.axis('off')
plt.savefig('D:/ai_engineering/week1_data_handling/numpy_image.png')
print("Image saved: numpy_image.png")

# ============================================
# CHALLENGE EXERCISES
# ============================================
print("\n" + "="*60)
print("CHALLENGE EXERCISES")
print("="*60)

print("\n1. Create a 5x5 checkerboard pattern (alternating 0s and 1s)")
# TODO: Implement this
checkerboard = np.zeros((5, 5))
checkerboard[::2, 1::2] = 1  # Hint: Use slicing
checkerboard[1::2, ::2] = 1
print(checkerboard)

print("\n2. Normalize an array to range [0, 1]")
data = np.array([10, 20, 30, 40, 50])
# Formula: (x - min) / (max - min)
normalized = (data - data.min()) / (data.max() - data.min())
print(f"Original: {data}")
print(f"Normalized: {normalized}")

print("\n3. Find indices of all elements > threshold")
arr = np.array([3, 7, 1, 9, 2, 8, 4])
threshold = 5
indices = np.where(arr > threshold)
print(f"Array: {arr}")
print(f"Indices where > {threshold}: {indices[0]}")
print(f"Values: {arr[indices]}")

print("\n" + "="*60)
print("NumPy exercises complete!")
print("Next: Run pandas_practice.py for Day 3-4")
print("="*60)