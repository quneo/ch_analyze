import numpy as np

xi = np.array([138, 139, 141, 143, 145, 147], dtype=float)
x0 = 142
M = 4

h = xi - x0

A = np.array([
    h**k for k in range(M + 1)
], dtype=float)

b = np.zeros(M + 1)
b[1] = 1

c = A.T @ np.linalg.solve(A @ A.T, b)

print("Коэффициенты:")
print(c)

print("Проверка условий:")
print(A @ c)
