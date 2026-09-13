#!/usr/bin/env python3
"""
generate_data.py — генерация тестовой СЛАУ Ax = b для метода Якоби.

Матрица делается диагонально доминирующей, чтобы гарантировать сходимость
метода Якоби. Точное решение x_true известно заранее (генерируется случайно),
после чего b = A @ x_true — это даёт дополнительный способ проверки помимо
numpy.linalg.solve.

Использование:
    python3 generate_data.py <n> [output_file] [seed]
"""
import sys
import numpy as np


def generate(n: int, filename: str, seed: int = 42, dominance: float = 2.0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))

    # Диагональное доминирование: |a_ii| > sum_{j!=i} |a_ij|
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        A[i, i] = row_sum * dominance + rng.uniform(1.0, 2.0)

    x_true = rng.uniform(-5.0, 5.0, size=n)
    b = A @ x_true

    with open(filename, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(" ".join(f"{v:.10f}" for v in A[i]) + "\n")
        f.write(" ".join(f"{v:.10f}" for v in b) + "\n")

    np.save(filename + ".A.npy", A)
    np.save(filename + ".b.npy", b)
    np.save(filename + ".x_true.npy", x_true)

    print(f"Сгенерирована СЛАУ размера n={n} -> {filename}")
    print(f"(также сохранены {filename}.A.npy, {filename}.b.npy, {filename}.x_true.npy для сверки)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    filename = sys.argv[2] if len(sys.argv) > 2 else "input.txt"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    generate(n, filename, seed)
