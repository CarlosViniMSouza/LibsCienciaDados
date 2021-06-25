import numpy as np

# Reading the examples codes:

# Here, we will have a Matrix 3x3
var = np.array([[1, 3, 9], [12, 15, 18], [21, 24, 27]])
print(var, "\n")

# Create an empty array with 2 elements:
var = np.empty(2)
print(var, "\n")

# Create an array with values that are spaced linearly in a specified interval:
var = np.linspace(1, 20, num = 40)
print(var, "\n")

# Adding, removing, and sorting elements:

# Method for Adding:
var1 = np.array([[1, 5, 10], [15, 20, 25]])
var2 = np.array([[3, 6, 9]])
var3 = np.concatenate((var1, var2), axis = 0)
print(var3, "\n")

# Method for Removing:
var4 = np.array([9, 8, 7, 6])
var4_2 = np.delete(var4, 3)
print(var4_2, "\n")

# Method for Sorting:
var5 = np.array([5, 10, 15, 20, 25])
var5_2 = np.sort(var5, axis = 0, kind='mergesort', order=None)
print(var5_2)
