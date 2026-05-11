import numpy as np


# Точка, в которой вычисляем построенный интерполяционный полином.
x0 = 1.3


def row_value(x, n):
    """Строка системы для условия p(x) = значение."""
    return [x**k for k in range(n + 1)]


def row_derivative(x, n):
    """Строка системы для условия p'(x) = значение."""
    return [0 if k == 0 else k * x**(k - 1) for k in range(n + 1)]


def row_second_derivative(x, n):
    """Строка системы для условия p''(x) = значение."""
    return [0 if k < 2 else k * (k - 1) * x**(k - 2) for k in range(n + 1)]


# В варианте 8 условий Эрмита, поэтому нужен полином степени 7.
n = 7

# Каждая строка M соответствует одному условию:
# f(0), f'(0), f''(0), f(1), f(2), f(3), f'(3), f''(3).
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

# Коэффициенты a0..a7 полинома p(x) = a0 + a1*x + ... + a7*x^7.
a = np.linalg.solve(M, b)
y = sum(a[k] * x0**k for k in range(n + 1))

print("Коэффициенты полинома:")
print(a)
print("p(1.3) =", y)
