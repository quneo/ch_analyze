import re
from pathlib import Path

import numpy as np


data_dir = Path(__file__).parent / "data" / "lab2"


def num_key(p: Path):
    m = re.search(r"\d+", p.stem)
    return int(m.group()) if m else 10**9


A_files = sorted(data_dir.glob("A*.txt"), key=num_key)
b_files = sorted(data_dir.glob("b*.txt"), key=num_key)

if len(A_files) == 0 or len(b_files) == 0:
    raise FileNotFoundError(f"Не найдены файлы A*.txt или b*.txt в папке {data_dir}.")

if len(A_files) != len(b_files):
    raise ValueError("Количество файлов A*.txt и b*.txt не совпадает.")

A_parts = [np.loadtxt(f, delimiter=",") for f in A_files]
b_parts = [np.loadtxt(f, delimiter=",").reshape(-1) for f in b_files]

A = np.vstack(A_parts)
b = np.concatenate(b_parts)

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

norm_res = np.linalg.norm(b - A @ x)

print("x =")
print(x)
print("\n||b - A x|| =", norm_res)
