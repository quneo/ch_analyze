import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Граница области устойчивости строится при xi = exp(i*phi), |xi| = 1.
phi = np.linspace(0, 2 * np.pi, 1000)
xi = np.exp(1j * phi)

# Полиномы rho и sigma задают конкретный разностный метод.
rho = 5 / 2 * xi**3 - 9 / 2 * xi + 2
sigma = 6 * xi**2 - 3 * xi

# z = h * lambda. Кривая z(phi) является границей области устойчивости.
z = rho / sigma

sns.set_theme()
plt.figure(figsize=(7, 6))
sns.lineplot(x=z.real, y=z.imag, sort=False, estimator=None)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True)
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Граница области устойчивости")
plt.axis("equal")
plt.tight_layout()
plt.show()
