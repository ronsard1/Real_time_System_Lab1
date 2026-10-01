"""Mini Project 1.2 - Measurement-based WCET estimation and confidence tool.

Runs the wcet_tool harness on four target functions of increasing structural
complexity, each with a random and a hand-crafted adversarial input strategy.

    python mini_project_1_2.py

Outputs: figures/hist_*.png, figures/summary_comparison.png, results_1_2.pdf
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import rts_common as rc      # noqa: E402
import wcet_tool as wt       # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
N = 1000                     # measurements per strategy
MARGIN_PCT = None            # None -> derived from the 65 % utilization budget
MS = 1000.0


# ============================================================ target functions
# 1. Early-exit branch ------------------------------------------------------
DATA = list(range(5000))


def early_exit_search(data, key):
    """Linear search that stops at the first match (early exit)."""
    for i, v in enumerate(data):
        if v == key:
            return i
    return -1


def gen_search_random(rng):
    return DATA, int(rng.integers(0, len(DATA)))       # key present, random position


def gen_search_adversarial(rng):
    return DATA, -1                                     # key absent -> full scan


# 2. Data-dependent loop count ---------------------------------------------
def collatz_steps(n):
    """Number of Collatz steps to reach 1: loop count depends on the data."""
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def gen_collatz_random(rng):
    return (int(rng.integers(1, 100000)),)


def gen_collatz_adversarial(rng):
    # 77031 has the longest Collatz chain (350 steps) of all n < 100000.
    return (77031,)


# 3. Nested conditionals ---------------------------------------------------
def classify_batch(values):
    """Per-value nested branches with different costs (heavy/medium/light)."""
    acc = 0
    for v in values:
        if v >= 0:
            if v % 2 == 0:
                if v % 3 == 0:
                    for j in range(20):          # heaviest path
                        acc += j * v
                else:
                    for j in range(6):
                        acc += j * v
            else:
                for j in range(2):
                    acc += j * v
        else:
            if v % 5 == 0:
                for j in range(4):
                    acc += j * v
            else:
                acc += v
    return acc


def gen_classify_random(rng):
    return (rng.integers(-1000, 1000, size=400).tolist(),)


def gen_classify_adversarial(rng):
    # every value >= 0, even and divisible by 3 -> heaviest branch each time
    return ((6 * rng.integers(1, 160, size=400)).tolist(),)


# 4. Fixed control flow (no data-dependent branches) -----------------------
COEFFS = [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05] * 2


def fir_filter(x):
    """Fixed-length FIR filter: the same operations for any input."""
    k = len(COEFFS)
    out = []
    for i in range(k, len(x)):
        s = 0.0
        for j in range(k):
            s += COEFFS[j] * x[i - j]
        out.append(s)
    return out


def gen_fir_random(rng):
    return (rng.normal(0, 1, 128).tolist(),)


def gen_fir_adversarial(rng):
    return ((1e6 * np.ones(128)).tolist(),)             # extreme values, same path


TARGETS = [
    ("early_exit_search", early_exit_search,
     {"random": gen_search_random, "adversarial (key absent)": gen_search_adversarial},
     "Early-exit branch"),
    ("collatz_steps", collatz_steps,
     {"random": gen_collatz_random, "adversarial (n = 77031)": gen_collatz_adversarial},
     "Data-dependent loop count"),
    ("classify_batch", classify_batch,
     {"random": gen_classify_random, "adversarial (all heavy branch)": gen_classify_adversarial},
     "Nested conditionals"),
    ("fir_filter", fir_filter,
     {"random": gen_fir_random, "adversarial (extreme values)": gen_fir_adversarial},
     "Fixed control flow"),
]


def main():
    results = {}
    for name, func, strategies, shape in TARGETS:
        res = wt.estimate(func, strategies, n=N, seed=1, margin_pct=MARGIN_PCT)
        results[name] = dict(res=res, shape=shape)
        wt.plot_histogram(res, name, os.path.join(FIG, f"hist_{name}.png"))
        s = res["strategies"]
        r, a = list(s.values())
        print(f"{name:18s} random max {r['stats']['max'] * MS:8.4f} ms | "
              f"adversarial max {a['stats']['max'] * MS:8.4f} ms | "
              f"margined {res['recommended'] * MS:8.4f} ms")

    # ---- summary chart
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [n for n, *_ in TARGETS]
    x = np.arange(len(names))
    rmax = [list(results[n]["res"]["strategies"].values())[0]["stats"]["max"] * MS for n in names]
    amax = [list(results[n]["res"]["strategies"].values())[1]["stats"]["max"] * MS for n in names]
    rec = [results[n]["res"]["recommended"] * MS for n in names]
    ax.bar(x - 0.27, rmax, 0.27, label="random max")
    ax.bar(x, amax, 0.27, label="adversarial max")
    ax.bar(x + 0.27, rec, 0.27, label="margined recommendation", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_yscale("log")
    ax.set_ylabel("Execution time (ms, log scale)")
    ax.set_title("Random vs adversarial vs margined WCET estimate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "summary_comparison.png"), dpi=200)
    plt.close(fig)

    # ---- PDF tables
    fmt = lambda v: f"{v * MS:.4f}"
    comp_rows, full_rows = [], []
    for n in names:
        res = results[n]["res"]
        (rn, r), (an, a) = list(res["strategies"].items())
        ratio = a["stats"]["max"] / r["stats"]["max"]
        comp_rows.append([n, results[n]["shape"], fmt(r["stats"]["max"]),
                          fmt(a["stats"]["max"]), f"{ratio:.2f}x",
                          "adversarial" if res["largest_from"] == an else "random",
                          fmt(res["recommended"])])
        for label, s in ((rn, r), (an, a)):
            st = s["stats"]
            full_rows.append([n, label, fmt(st["mean"]), fmt(st["std"]), fmt(st["min"]),
                              fmt(st["max"]), fmt(st["p90"]), fmt(st["p95"]), fmt(st["p99"]),
                              f"{st['max_over_p99']:.2f}", "YES" if st["outlier_flag"] else "no"])
    util_rows = [[f"{u * 100:.0f}%", f"+{wt.margin_from_utilization(u) * 100:.0f}%"]
                 for u in (0.6, 0.65, 0.7)]
    sections = [
        dict(heading="1. Comparison: random vs adversarial vs margined (ms)",
             notes=[f"{N} measurements per strategy, each call timed with time.perf_counter(). "
                    f"Margined recommendation = overall observed maximum x (1 + margin); default "
                    f"margin +{100 * wt.margin_from_utilization():.0f}% (65% utilization budget).",
                    wt.LIMITATIONS],
             header=["Function", "Control-flow shape", "Random max", "Adversarial max",
                     "Adv / random", "Largest sample from", "Margined estimate"],
             rows=comp_rows),
        dict(heading="2. Full statistics per strategy (ms)",
             notes=[f"Outlier flag: YES when max > {wt.OUTLIER_RATIO} x the 99th percentile "
                    "(a rare spike worth investigating, often OS noise rather than the code path)."],
             header=["Function", "Strategy", "Mean", "Std", "Min", "Max", "p90", "p95", "p99",
                     "Max/p99", "Outlier"],
             rows=full_rows),
        dict(heading="3. Safety margin versus utilization budget",
             notes=["Keeping utilization below a budget U is equivalent to inflating C by 1/U - 1 "
                    "(the 60-70% convention of Section 1.7). margin_pct in wcet_tool.estimate() "
                    "overrides this default."],
             header=["Utilization budget", "Equivalent margin on C"], rows=util_rows),
    ]
    rc.save_tables_pdf(os.path.join(HERE, "results_1_2.pdf"),
                       "Mini Project 1.2: results", sections)

    if os.environ.get("RTS_JSON_OUT"):
        dump = {}
        for n in names:
            res = results[n]["res"]
            dump[n] = dict(shape=results[n]["shape"], recommended=res["recommended"],
                           overall_max=res["overall_max"], largest_from=res["largest_from"],
                           strategies={k: v["stats"] for k, v in res["strategies"].items()})
        with open(os.environ["RTS_JSON_OUT"], "w") as fh:
            json.dump(dump, fh, indent=1, default=float)
    print("Done. Figures in ./figures, tables in results_1_2.pdf")


if __name__ == "__main__":
    main()
