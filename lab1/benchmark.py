#!/usr/bin/env python3
"""
benchmark.py — исследование зависимости времени выполнения метода Якоби
от объёма задачи (размера СЛАУ n).

Для каждого n из заданного списка:
  1. генерирует тестовую СЛАУ (generate_data.py::generate)
  2. запускает ./jacobi и парсит время/итерации/невязку из output-файла
  3. запускает verify.py для проверки корректности
  4. сохраняет результат в results.csv

По окончании строит график time_ms(n) в results_plot.png (если установлен
matplotlib).

Использование:
    python3 benchmark.py                       # размеры по умолчанию
    python3 benchmark.py 100 500 1000 2000 4000 8000
"""
import sys
import os
import csv
import subprocess
import shutil

from generate_data import generate


def parse_meta(output_file):
    meta = {}
    with open(output_file) as f:
        f.readline()  # n
        f.readline()  # x values
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0]] = float(parts[1])
    return meta


def run_case(n, eps=1e-8, max_iter=100000, workdir="bench_tmp"):
    os.makedirs(workdir, exist_ok=True)
    input_file = os.path.join(workdir, f"input_{n}.txt")
    output_file = os.path.join(workdir, f"output_{n}.txt")

    generate(n, input_file, seed=42)

    jacobi_bin = "./jacobi" if os.path.exists("./jacobi") else "jacobi"
    result = subprocess.run(
        [jacobi_bin, input_file, output_file, str(eps), str(max_iter)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[n={n}] ОШИБКА запуска jacobi:\n{result.stderr}")
        return None

    verify = subprocess.run(
        [sys.executable, "verify.py", input_file, output_file],
        capture_output=True, text=True
    )
    verified = (verify.returncode == 0)

    meta = parse_meta(output_file)
    meta["n"] = n
    meta["verified"] = verified
    return meta


def main():
    sizes = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else \
        [100, 250, 500, 1000, 2000, 4000, 8000]

    if not (os.path.exists("./jacobi") or shutil.which("jacobi")):
        print("Не найден исполняемый файл jacobi. Сначала выполните: make")
        sys.exit(1)

    results = []
    for n in sizes:
        print(f"Запуск для n = {n} ...")
        meta = run_case(n)
        if meta:
            results.append(meta)
            status = "OK" if meta["verified"] else "FAILED"
            print(f"  time_ms={meta['time_ms']:.4f}  iterations={int(meta['iterations'])}  "
                  f"residual={meta['residual']:.3e}  verify={status}")

    # CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "time_ms", "iterations", "residual", "eps", "verified"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print("\nРезультаты сохранены в results.csv")

    # График (опционально)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [r["n"] for r in results]
        times = [r["time_ms"] for r in results]

        plt.figure(figsize=(7, 5))
        plt.plot(ns, times, marker="o")
        plt.xlabel("Размер задачи n")
        plt.ylabel("Время выполнения, мс")
        plt.title("Зависимость времени выполнения метода Якоби от размера задачи")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results_plot.png", dpi=150)
        print("График сохранён в results_plot.png")
    except ImportError:
        print("matplotlib не установлен — график пропущен (pip install matplotlib)")


if __name__ == "__main__":
    main()
