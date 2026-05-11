import sympy as sp


# Ищем коэффициенты линейного многошагового метода для y' = f(x, y).
a1, a2, a3, a4, a5, a6, a7 = sp.symbols("a1 a2 a3 a4 a5 a6 a7")

# alpha отвечает за значения y, beta - за значения правой части f.
alpha = {
    2: a1,
    1: a2,
    0: a3,
    -1: a4
}

beta = {
    1: a5,
    0: a6,
    -1: a7
}

# Эти коэффициенты заданы вариантом.
values = {
    a2: 0,
    a4: 2,
    a7: 0
}

unknowns = [a1, a3, a5, a6]
equations = []

# Первое условие согласованности: сумма alpha_j должна быть равна нулю.
equations.append(
    sp.Eq(sum(alpha[j] for j in alpha), 0)
)

# Условия порядка: схема должна точно работать на степенных функциях.
for k in range(1, 8):
    left = sum(alpha[j] * j**k for j in alpha)
    right = k * sum(beta[j] * j**(k - 1) for j in beta)
    equations.append(sp.Eq(left, right))

max_order = -1
solution = None

# Добавляем условия по одному, пока система еще совместна.
for p in range(len(equations)):
    eqs = [eq.subs(values) for eq in equations[:p + 1]]
    sol = sp.solve(eqs, unknowns, dict=True)

    if sol:
        max_order = p
        solution = sol[0]
    else:
        break

print("Максимальный порядок:", max_order)

for var in [a1, a2, a3, a4, a5, a6, a7]:
    if var in values:
        print(var, "=", values[var])
    else:
        print(var, "=", sp.simplify(solution[var]))
