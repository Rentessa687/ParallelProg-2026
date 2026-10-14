#!/usr/bin/env python3
"""
verify.py — автоматизированная верификация результата C++-программы.

Читает исходную СЛАУ из input_file и решение из output_file (полученное
методом Якоби), независимо решает ту же систему через numpy.linalg.solve
и сравнивает решения. Также, если рядом лежат *.x_true.npy (сгенерированные
generate_data.py), сравнивает с точным решением.

Использование:
    python3 verify.py <input_file> <output_file>

Код возврата: 0 — верификация пройдена, 1 — расхождение превышает допуск.
"""
import sys
import os
import numpy as np


def read_matrix_input(filename):
    with open(filename) as f:
        n = int(f.readline())
        A = np.zeros((n, n))
        for i in range(n):
            A[i] = list(map(float, f.readline().split()))
        b = np.array(list(map(float, f.readline().split())))
    return n, A, b


def read_output(filename):
    with open(filename) as f:
        n = int(f.readline())
        x = np.array(list(map(float, f.readline().split())))
        meta = {}
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                key, val = parts
                meta[key] = float(val)
    return n, x, meta


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 verify.py <input_file> <output_file>")
        sys.exit(2)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    n_in, A, b = read_matrix_input(input_file)
    n_out, x_program, meta = read_output(output_file)

    if n_in != n_out:
        print(f"ОШИБКА: несовпадение размерности: input n={n_in}, output n={n_out}")
        sys.exit(1)

    # Эталонное решение через NumPy (LAPACK)
    x_numpy = np.linalg.solve(A, b)

    abs_err = np.abs(x_program - x_numpy)
    rel_err = np.linalg.norm(x_program - x_numpy) / np.linalg.norm(x_numpy)
    max_abs_err = np.max(abs_err)

    print("=== Отчёт о верификации ===")
    print(f"Размер задачи n          : {n_in}")
    print(f"Число итераций Якоби     : {int(meta.get('iterations', -1))}")
    print(f"Время выполнения, мс     : {meta.get('time_ms', -1):.4f}")
    print(f"Невязка ||Ax-b|| (из C++): {meta.get('residual', -1):.6e}")
    print(f"Макс. абс. ошибка vs NumPy : {max_abs_err:.6e}")
    print(f"Относительная L2-ошибка  : {rel_err:.6e}")

    # Если есть точное решение, использованное при генерации данных — сверим и с ним
    x_true_path = input_file + ".x_true.npy"
    if os.path.exists(x_true_path):
        x_true = np.load(x_true_path)
        rel_err_true = np.linalg.norm(x_program - x_true) / np.linalg.norm(x_true)
        print(f"Относительная ошибка vs x_true: {rel_err_true:.6e}")

    tol = 1e-4
    print(f"\nДопуск на относительную ошибку: {tol:.1e}")
    if rel_err < tol:
        print("РЕЗУЛЬТАТ: OK — решение верифицировано")
        sys.exit(0)
    else:
        print("РЕЗУЛЬТАТ: FAILED — расхождение превышает допуск")
        sys.exit(1)


if __name__ == "__main__":
    main()
