
import numpy as np


# 1. Create NumPy arrays
print("=== 1. NumPy Arrays ===")

arr1 = np.array([10, 20, 30, 40, 50])
arr2 = np.array([2, 4, 5, 8, 10])

print("Array 1:", arr1)
print("Array 2:", arr2)

print("Shape:", arr1.shape)
print("Dimensions:", arr1.ndim)
print("Data type:", arr1.dtype)


# 2. Arithmetic operations
print("\n=== 2. Arithmetic Operations ===")

print("Addition:", arr1 + arr2)
print("Subtraction:", arr1 - arr2)
print("Multiplication:", arr1 * arr2)
print("Division:", arr1 / arr2)


# 3. Array slicing
print("\n=== 3. Array Slicing ===")

numbers = np.array([10, 20, 30, 40, 50, 60])

print("Original array:", numbers)
print("First three elements:", numbers[:3])
print("Elements from index 2 to 4:", numbers[2:5])
print("Every second element:", numbers[::2])
print("Last two elements:", numbers[-2:])


# 4. Reshaping arrays
print("\n=== 4. Reshaping Arrays ===")

values = np.arange(1, 13)

matrix = values.reshape(3, 4)

print("Original 1D array:", values)
print("Reshaped 3x4 array:\n", matrix)
print("Matrix shape:", matrix.shape)
print("Matrix dimensions:", matrix.ndim)


# 5. Broadcasting
print("\n=== 5. Broadcasting ===")

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

addition_values = np.array([1, 2, 3])

print("Original matrix:\n", matrix)
print("Values added to each row:", addition_values)
print("Broadcasting result:\n", matrix + addition_values)


# 6. More useful NumPy operations
print("\n=== 6. Numerical Operations ===")

data = np.array([10, 20, 30, 40, 50])

print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
print("Mean:", np.mean(data))
print("Sum:", np.sum(data))

print("\nDay 41 NumPy practice completed successfully!")
