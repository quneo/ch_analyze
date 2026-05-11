import sympy as sp


# Решаем линейную неоднородную рекурсию второго порядка.
n = sp.symbols("n", integer=True, nonnegative=True)
y = sp.Function("y")

recurrence = sp.Eq(
    y(n) - 2 * y(n - 1) + y(n - 2),
    2**n
)

# Начальные условия позволяют выбрать единственное решение рекурсии.
solution = sp.rsolve(
    recurrence,
    y(n),
    {
        y(0): 2,
        y(1): 2
    }
)

print("Явная формула:")
print(sp.simplify(solution))

print("\nПервые значения последовательности:")
for k in range(10):
    print(f"y_{k} =", solution.subs(n, k))
