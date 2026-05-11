import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

omega = 1.0
g = 1.0

a = 0.0
b = np.pi / 2

u_a = 2.0
u_b = 3.0

def exact_solution(t):
    L = b - a
    C1 = u_a - g / omega**2
    C2 = (u_b - g / omega**2 - C1 * np.cos(omega * L)) / np.sin(omega * L)
    return g / omega**2 + C1 * np.cos(omega * t) + C2 * np.sin(omega * t)

def finite_difference_solution(n):
    t = np.linspace(a, b, n)
    h = (b - a) / (n - 1)

    A = np.zeros((n - 2, n - 2))
    F = np.full(n - 2, g * h**2)

    for i in range(n - 2):
        A[i, i] = -2 + omega**2 * h**2

        if i > 0:
            A[i, i - 1] = 1

        if i < n - 3:
            A[i, i + 1] = 1

    F[0] -= u_a
    F[-1] -= u_b

    u_inner = np.linalg.solve(A, F)

    u = np.zeros(n)
    u[0] = u_a
    u[-1] = u_b
    u[1:-1] = u_inner

    return t, u

def runge_error(n):
    t1, u1 = finite_difference_solution(n)
    t2, u2 = finite_difference_solution(2 * n - 1)

    u2_on_t1 = u2[::2]

    p = 2
    err = np.abs(u2_on_t1 - u1) / (2**p - 1)

    return np.max(err), err

n_values = [5, 9, 17, 33, 65]

print("n      h              max true error       Runge error")

for n in n_values:
    t, u_num = finite_difference_solution(n)
    u_ex = exact_solution(t)

    h = (b - a) / (n - 1)
    true_error = np.max(np.abs(u_ex - u_num))
    runge, _ = runge_error(n)

    print(f"{n:<6} {h:<14.8f} {true_error:<20.10e} {runge:<20.10e}")

n = 33

t, u_num = finite_difference_solution(n)
u_ex = exact_solution(t)

tt = np.linspace(a, b, 500)
uu = exact_solution(tt)

sns.set_theme()
plt.figure(figsize=(8, 5))
sns.lineplot(x=tt, y=uu)
plt.grid(True)
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Точное решение")
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(x=t, y=u_num, marker="o")
plt.grid(True)
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Численное решение")
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(x=t, y=np.abs(u_ex - u_num), marker="o")
plt.grid(True)
plt.xlabel("t")
plt.ylabel("Ошибка")
plt.title("Погрешность численного решения")
plt.show()
