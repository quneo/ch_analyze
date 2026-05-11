import numpy as np


def lu_full_pivot(A, tol=1e-12):
    """LU-разложение с полным выбором главного элемента."""
    A = A.astype(float).copy()
    n = A.shape[0]
    p = np.arange(n)
    q = np.arange(n)
    L = np.eye(n)
    U = A.copy()

    for k in range(n):
        sub = np.abs(U[k:, k:])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i = k + i_rel
        j = k + j_rel

        if abs(U[i, j]) < tol:
            raise ValueError("Матрица вырождена")

        # Перестановка строк повышает численную устойчивость.
        if i != k:
            U[[k, i], :] = U[[i, k], :]
            L[[k, i], :k] = L[[i, k], :k]
            p[[k, i]] = p[[i, k]]

        # Перестановку столбцов запоминаем, чтобы потом восстановить порядок x.
        if j != k:
            U[:, [k, j]] = U[:, [j, k]]
            q[[k, j]] = q[[j, k]]

        # Обнуляем элементы ниже диагонали и заполняем матрицу L.
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return L, U, p, q


def forward_substitution(L, b):
    """Решает L y = b для нижнетреугольной матрицы L."""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def backward_substitution(U, y):
    """Решает U x = y для верхнетреугольной матрицы U."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def solve_lu_full_pivot(A, b):
    """Решает систему A x = b через LU с полным выбором главного элемента."""
    L, U, p, q = lu_full_pivot(A)
    y = forward_substitution(L, b[p])
    z = backward_substitution(U, y)
    x = np.zeros_like(z)
    x[q] = z
    return x


A = np.array([
    [0.440241, 0.923916, 0.834289, 0.143245, 1.000316],
    [1.273960, 1.665023, 0.748191, 0.340312, 0.366482],
    [-0.435433, 2.149114, -0.809408, -0.496640, 1.633513],
    [0.101153, 1.499158, 2.016233, -0.607518, 0.229853]
], dtype=float)

b = np.array([0.128352, -2.133823, -0.987770, -1.389915], dtype=float)

C = np.array([
    [0.000000, -1.038847, -1.061306, 0.000000, -0.892186],
    [0.000000, -0.305082, 1.022882, -0.448738, 0.000000],
    [0.368302, 0.000000, 0.825692, -1.119023, 0.000000]
], dtype=float)

d = np.array([0.779833, -0.495342, 2.150026], dtype=float)

# МНК с ограничениями C x = d сводится к системе ККТ.
# В этой лабораторной систему ККТ решаем своим LU-разложением.
KKT = np.block([
    [A.T @ A, C.T],
    [C, np.zeros((C.shape[0], C.shape[0]))]
])

rhs = np.concatenate([A.T @ b, d])
sol = solve_lu_full_pivot(KKT, rhs)
x = sol[:A.shape[1]]

print("x =", x)
print("C x =", C @ x)
print("d =", d)
print("||b - A x|| =", np.linalg.norm(b - A @ x))
