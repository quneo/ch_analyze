import numpy as np

x0 = 1.3

def row_value(x, n):
    return [x**k for k in range(n + 1)]

def row_derivative(x, n):
    return [0 if k == 0 else k * x**(k - 1) for k in range(n + 1)]

def row_second_derivative(x, n):
    return [0 if k < 2 else k * (k - 1) * x**(k - 2) for k in range(n + 1)]

n = 7

M = np.array([
    row_value(0, n),
    row_derivative(0, n),
    row_second_derivative(0, n),
    row_value(1, n),
    row_value(2, n),
    row_value(3, n),
    row_derivative(3, n),
    row_second_derivative(3, n)
], dtype=float)

b = np.array([1, -2, 1, 0, -2, 1, -2, 1], dtype=float)

a = np.linalg.solve(M, b)

y = sum(a[k] * x0**k for k in range(n + 1))

print("y =", y)
print(M)
