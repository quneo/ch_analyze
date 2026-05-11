import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.legendre import legfit, legval


sns.set_theme()

# ---------- Часть 1. Интерполяция в базисе полиномов Лежандра ----------

a, b = 0.3, 1.3
x_nodes = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3], dtype=float)
y_nodes = np.exp(-x_nodes**2 / 2)


def to_legendre_interval(x):
    """Переводит отрезок [a, b] в стандартный отрезок [-1, 1]."""
    return (2 * x - (a + b)) / (b - a)


m = 5
A = np.zeros((len(x_nodes), m + 1))

# A[:, k] - значения k-го полинома Лежандра в узлах.
for k in range(m + 1):
    c = np.zeros(k + 1)
    c[-1] = 1
    A[:, k] = legval(to_legendre_interval(x_nodes), c)

alpha = np.linalg.solve(A, y_nodes)

x = np.linspace(a, b, 500)
B = np.zeros((len(x), m + 1))

for k in range(m + 1):
    c = np.zeros(k + 1)
    c[-1] = 1
    B[:, k] = legval(to_legendre_interval(x), c)

p = B @ alpha
psi = np.exp(-x**2 / 2)
diff = psi - p

print("Коэффициенты интерполяционного полинома:")
print(alpha)

# ---------- Часть 2. Сравнение МНК и встроенного legfit ----------

x_nodes_fit = np.linspace(-1, 1, 21)


def f(x):
    return np.exp(-x**2 / 2)


A_fit = np.zeros((len(x_nodes_fit), m + 1))

for k in range(m + 1):
    c = np.zeros(k + 1)
    c[-1] = 1
    A_fit[:, k] = legval(x_nodes_fit, c)

# Матрица перехода от значений функции в узлах к коэффициентам МНК.
M = np.linalg.inv(A_fit.T @ A_fit) @ A_fit.T

y_fit = f(x_nodes_fit)
alpha1 = M @ y_fit
alpha2 = legfit(x_nodes_fit, y_fit, m)

print("\nМатрица M:")
print(M)
print("\nКоэффициенты через M:")
print(alpha1)
print("\nКоэффициенты через legfit:")
print(alpha2)
print("\nРазность коэффициентов:")
print(alpha1 - alpha2)

x_fit = np.linspace(-1, 1, 500)
p1 = legval(x_fit, alpha1)
p2 = legval(x_fit, alpha2)

# Оба графика показываем в одном окне.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.lineplot(x=x, y=diff, ax=axes[0])
axes[0].grid(True)
axes[0].set_xlabel("x")
axes[0].set_ylabel("psi(x) - p(x)")
axes[0].set_title("Ошибка интерполяции")

sns.lineplot(x=x_fit, y=p1 - p2, ax=axes[1])
axes[1].grid(True)
axes[1].set_xlabel("x")
axes[1].set_ylabel("p_M(x) - p_fit(x)")
axes[1].set_title("Разность двух аппроксимаций")

plt.tight_layout()
plt.show()
