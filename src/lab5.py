import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.legendre import legval

a, b = 0.3, 1.3
x_nodes = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3], dtype=float)
y_nodes = np.exp(-x_nodes**2 / 2)

def t(x):
    return (2 * x - (a + b)) / (b - a)

m = 5
A = np.zeros((len(x_nodes), m + 1))
for k in range(m + 1):
    c = np.zeros(k + 1)
    c[-1] = 1
    A[:, k] = legval(t(x_nodes), c)

alpha = np.linalg.solve(A, y_nodes)

x = np.linspace(a, b, 500)
B = np.zeros((len(x), m + 1))
for k in range(m + 1):
    c = np.zeros(k + 1)
    c[-1] = 1
    B[:, k] = legval(t(x), c)

p = B @ alpha
psi = np.exp(-x**2 / 2)
diff = psi - p

print(alpha)

sns.set_theme()
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=diff)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("psi(x) - p(x)")
plt.title("Разность функции и аппроксимирующего полинома")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.legendre import legval, legfit

x_nodes = np.linspace(-1, 1, 21)

def f(x):
    return np.exp(-x**2 / 2)

m = 5
A = np.zeros((len(x_nodes), m + 1))
for k in range(m + 1):
    c = np.zeros(k + 1)
    c[-1] = 1
    A[:, k] = legval(x_nodes, c)

M = np.linalg.inv(A.T @ A) @ A.T

y = f(x_nodes)
alpha1 = M @ y
alpha2 = legfit(x_nodes, y, m)

print("M =")
print(M)
print()
print("Коэффициенты через M:")
print(alpha1)
print()
print("Коэффициенты через legfit:")
print(alpha2)
print()
print("Разность коэффициентов:")
print(alpha1 - alpha2)

x = np.linspace(-1, 1, 500)
p1 = legval(x, alpha1)
p2 = legval(x, alpha2)

plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=p1 - p2)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("p_M(x) - p_fit(x)")
plt.title("Разность двух аппроксимаций")
plt.show()
