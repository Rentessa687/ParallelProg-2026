#!/usr/bin/env python3
"""
benchmark_omp.py — исследование OpenMP-версии метода Якоби.

Для каждого размера n:
  1. генерирует СЛАУ (generate_data.py::generate);
  2. замеряет последовательную версию ./jacobi_seq;
  3. замеряет ./jacobi_omp для каждого числа потоков;
  4. каждую конфигурацию запускает reps раз (после одного прогревочного
     запуска) и берёт медиану времени;
  5. проверяет результат через verify.py (NumPy/LAPACK).

Результаты: results.csv и results_plot.png.

Использование:
    python3 benchmark_omp.py
    python3 benchmark_omp.py --sizes 200 400 800 --threads 1 2 4 --reps 5
    python3 benchmark_omp.py --dominance 1.05 --eps 1e-10 --out results_heavy.csv --plot results_heavy.png
"""
import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys

from generate_data import generate

SEQ_BIN = "./jacobi_seq"
OMP_BIN = "./jacobi_omp"


def parse_meta(output_file):
    meta = {}
    with open(output_file) as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0]] = float(parts[1])
    return meta


def run_once(binary, input_file, output_file, eps, max_iter, threads):
    cmd = [binary, input_file, output_file, str(eps), str(max_iter)]
    if threads is not None:
        cmd.append(str(threads))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка запуска {' '.join(cmd)}:\n{result.stderr}")
    return parse_meta(output_file)


def verify(input_file, output_file):
    result = subprocess.run(
        [sys.executable, "verify.py", input_file, output_file],
        capture_output=True, text=True
    )
    return result.returncode == 0


def measure(binary, input_file, output_file, eps, max_iter, threads, reps):
    run_once(binary, input_file, output_file, eps, max_iter, threads)
    times = []
    meta = None
    for _ in range(reps):
        meta = run_once(binary, input_file, output_file, eps, max_iter, threads)
        times.append(meta["time_ms"])
    return {
        "time_ms": statistics.median(times),
        "time_min_ms": min(times),
        "iterations": int(meta["iterations"]),
        "residual": meta["residual"],
        "verified": verify(input_file, output_file),
    }


def make_plots(rows, sizes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — график пропущен (pip install matplotlib)")
        return

    omp = [r for r in rows if r["version"] == "omp"]
    seq = [r for r in rows if r["version"] == "seq"]
    thread_values = sorted({r["threads"] for r in omp})

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    for n in sizes:
        pts = sorted((r for r in omp if r["n"] == n), key=lambda r: r["threads"])
        ts = [r["threads"] for r in pts]
        ax[0][0].plot(ts, [r["time_ms"] for r in pts], marker="o", label=f"n={n}")
        ax[0][1].plot(ts, [r["speedup"] for r in pts], marker="o", label=f"n={n}")
        ax[1][0].plot(ts, [r["efficiency"] for r in pts], marker="o", label=f"n={n}")

    ax[0][0].set_yscale("log")
    ax[0][0].set_xlabel("Число потоков")
    ax[0][0].set_ylabel("Время выполнения, мс")
    ax[0][0].set_title("Время выполнения от числа потоков")

    ax[0][1].plot(thread_values, thread_values, "k--", label="идеальное")
    ax[0][1].set_xlabel("Число потоков")
    ax[0][1].set_ylabel("Ускорение T_seq / T_p")
    ax[0][1].set_title("Ускорение")

    ax[1][0].axhline(1.0, color="k", linestyle="--")
    ax[1][0].set_xlabel("Число потоков")
    ax[1][0].set_ylabel("Эффективность S_p / p")
    ax[1][0].set_title("Эффективность")

    seq_pts = sorted(seq, key=lambda r: r["n"])
    ax[1][1].plot([r["n"] for r in seq_pts], [r["time_ms"] for r in seq_pts],
                  marker="s", color="k", label="последовательная")
    for p in thread_values:
        pts = sorted((r for r in omp if r["threads"] == p), key=lambda r: r["n"])
        ax[1][1].plot([r["n"] for r in pts], [r["time_ms"] for r in pts],
                      marker="o", label=f"p={p}")
    ax[1][1].set_yscale("log")
    ax[1][1].set_xlabel("Размер задачи n")
    ax[1][1].set_ylabel("Время выполнения, мс")
    ax[1][1].set_title("Время выполнения от размера задачи")

    for a in (ax[0][0], ax[0][1], ax[1][0]):
        a.set_xticks(thread_values)

    for a in ax.flat:
        a.grid(True)
        a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"График сохранён в {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=100000)
    parser.add_argument("--dominance", type=float, default=2.0)
    parser.add_argument("--workdir", default="bench_tmp")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--plot", default="results_plot.png")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    for binary in (SEQ_BIN, OMP_BIN):
        if not os.path.exists(binary):
            print(f"Не найден {binary}. Сначала выполните: make")
            sys.exit(1)

    os.makedirs(args.workdir, exist_ok=True)
    rows = []

    for n in args.sizes:
        print(f"\n=== n = {n} ===")
        input_file = os.path.join(args.workdir, f"input_{n}.txt")
        output_file = os.path.join(args.workdir, f"output_{n}.txt")
        generate(n, input_file, seed=42, dominance=args.dominance)

        seq = measure(SEQ_BIN, input_file, output_file, args.eps, args.max_iter, None, args.reps)
        t_seq = seq["time_ms"]
        rows.append({"n": n, "version": "seq", "threads": 1, **seq,
                     "eps": args.eps, "speedup": 1.0, "efficiency": 1.0})
        print(f"  seq        : {t_seq:10.4f} мс  iters={seq['iterations']}  "
              f"verify={'OK' if seq['verified'] else 'FAILED'}")

        for p in args.threads:
            res = measure(OMP_BIN, input_file, output_file, args.eps, args.max_iter, p, args.reps)
            speedup = t_seq / res["time_ms"]
            rows.append({"n": n, "version": "omp", "threads": p, **res,
                         "eps": args.eps, "speedup": speedup, "efficiency": speedup / p})
            print(f"  omp p={p:<3}  : {res['time_ms']:10.4f} мс  iters={res['iterations']}  "
                  f"S={speedup:6.2f}  E={speedup / p:5.2f}  "
                  f"verify={'OK' if res['verified'] else 'FAILED'}")

        if not args.keep:
            for f in glob.glob(os.path.join(args.workdir, f"*_{n}.txt*")):
                os.remove(f)

    fields = ["n", "version", "threads", "time_ms", "time_min_ms", "iterations",
              "residual", "eps", "speedup", "efficiency", "verified"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nРезультаты сохранены в {args.out}")

    make_plots(rows, args.sizes, args.plot)


if __name__ == "__main__":
    main()
