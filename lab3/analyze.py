import csv
import glob
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

times = defaultdict(list)
for f in sorted(glob.glob("results/res_*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            times[(int(r["n"]), int(r["procs"]))].append(float(r["time_ms"]))

sizes = sorted({n for n, _ in times})
procs = sorted({p for _, p in times})

rows = []
for n in sizes:
    t1 = st.mean(times[(n, 1)])
    for p in procs:
        t = st.mean(times[(n, p)])
        rows.append({
            "n": n,
            "procs": p,
            "runs": len(times[(n, p)]),
            "time_ms": round(t, 4),
            "time_min_ms": round(min(times[(n, p)]), 4),
            "time_max_ms": round(max(times[(n, p)]), 4),
            "speedup": round(t1 / t, 3),
            "efficiency": round(t1 / t / p, 3),
        })

with open("results_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

def table(key):
    print(f"\n{key}")
    print("n\\p".ljust(7) + "".join(str(p).rjust(10) for p in procs))
    for n in sizes:
        vals = [next(r[key] for r in rows if r["n"] == n and r["procs"] == p) for p in procs]
        print(str(n).ljust(7) + "".join(f"{v:10.3f}" for v in vals))

for k in ("time_ms", "speedup", "efficiency"):
    table(k)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for n in sizes:
    sub = [r for r in rows if r["n"] == n]
    ax[0].plot(procs, [r["time_ms"] for r in sub], marker="o", label=f"n={n}")
    ax[1].plot(procs, [r["speedup"] for r in sub], marker="o", label=f"n={n}")
    ax[2].plot(procs, [r["efficiency"] for r in sub], marker="o", label=f"n={n}")
ax[1].plot(procs, procs, "k--", label="идеальное")
ax[0].set_yscale("log")
ax[0].set_title("Время выполнения")
ax[0].set_ylabel("мс")
ax[1].set_title("Ускорение")
ax[2].set_title("Эффективность")
for a in ax:
    a.set_xlabel("число процессов")
    a.set_xscale("log", base=2)
    a.set_xticks(procs)
    a.set_xticklabels(procs)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results_plot.png", dpi=150)
