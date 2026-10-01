"""Measurement-based WCET estimation tool (Mini Project 1.2).

Works with any target function and any input generator. It reports the
measured statistics, a margined value for scheduling, and an explicit
statement of the estimate's limitations, never a single unqualified number.
"""
import time

import numpy as np

OUTLIER_RATIO = 1.5          # flag if max > OUTLIER_RATIO * 99th percentile
DEFAULT_UTILIZATION = 0.65   # 60-70 % utilization-budget convention (Section 1.7)


# ------------------------------------------------------------ measurement
def measure(func, inputs, warmup=20):
    """Time func(*args) once per input with time.perf_counter().

    `inputs` is a list (or any iterable) of argument tuples. Every duration is
    kept. Returns (durations_seconds as a numpy array, inputs as a list)."""
    inputs = list(inputs)
    for args in inputs[:warmup]:                 # warm caches / interpreter state
        func(*args)
    durations = np.zeros(len(inputs))
    for i, args in enumerate(inputs):
        t0 = time.perf_counter()
        func(*args)
        durations[i] = time.perf_counter() - t0
    return durations, inputs


def campaign(func, input_generator, n, seed=0):
    """Draw n inputs from input_generator(rng) and measure them."""
    rng = np.random.default_rng(seed)
    inputs = [input_generator(rng) for _ in range(n)]
    return measure(func, inputs)


# -------------------------------------------------------------- statistics
def statistics(durations):
    d = np.asarray(durations)
    s = dict(n=int(d.size), mean=float(d.mean()), std=float(d.std(ddof=1)),
             min=float(d.min()), max=float(d.max()),
             p90=float(np.percentile(d, 90)), p95=float(np.percentile(d, 95)),
             p99=float(np.percentile(d, 99)))
    s["outlier_flag"] = bool(s["max"] > OUTLIER_RATIO * s["p99"])
    s["max_over_p99"] = s["max"] / s["p99"]
    return s


# ---------------------------------------------------------- safety margin
def margin_from_utilization(budget=DEFAULT_UTILIZATION):
    """Margin (fraction) equivalent to keeping utilization under `budget`:
    inflating C by 1/budget - 1 (e.g. 0.65 -> +54 %)."""
    return 1.0 / budget - 1.0


def margined_wcet(observed_max, margin_pct=None, utilization=DEFAULT_UTILIZATION):
    """Recommended WCET-for-scheduling = observed max x (1 + margin).
    margin_pct overrides the utilization-derived default."""
    margin = margin_pct / 100.0 if margin_pct is not None else margin_from_utilization(utilization)
    return observed_max * (1.0 + margin)


# ------------------------------------------------------------ full report
def estimate(func, strategies, n=1000, seed=0, margin_pct=None,
             utilization=DEFAULT_UTILIZATION):
    """Run every strategy {name: input_generator} and summarise.

    Returns dict with per-strategy durations/inputs/stats and the combined
    recommendation (max over all strategies x margin)."""
    res = {"strategies": {}}
    overall_max = 0.0
    for name, gen in strategies.items():
        d, inputs = campaign(func, gen, n, seed)
        res["strategies"][name] = dict(durations=d, inputs=inputs, stats=statistics(d))
        overall_max = max(overall_max, float(d.max()))
    res["overall_max"] = overall_max
    res["recommended"] = margined_wcet(overall_max, margin_pct, utilization)
    res["margin_pct"] = (margin_pct if margin_pct is not None
                         else 100.0 * margin_from_utilization(utilization))
    names = list(res["strategies"])
    res["largest_from"] = max(
        names, key=lambda k: res["strategies"][k]["stats"]["max"])
    return res


LIMITATIONS = (
    "Measured, not proven: no test campaign guarantees the true worst-case path, "
    "cache/pipeline state or interference was exercised. The margin is a heuristic "
    "and cannot turn a measurement into a sound bound; a static WCET analysis (or "
    "hybrid method) is needed for certification.")


def plot_histogram(res, name, path, unit_ms=1000.0):
    """Histogram per strategy with each strategy's measured maximum (dashed)
    and the margined estimate (solid red). The x-axis always spans up to the
    margined estimate so none of the markers is clipped; counts use a log
    scale so rare long samples remain visible."""
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    rec = res["recommended"] * unit_ms
    allv = np.concatenate([s["durations"] for s in res["strategies"].values()]) * unit_ms
    bins = np.linspace(allv.min() * 0.95, rec * 1.05, 90)
    for (sname, s), c in zip(res["strategies"].items(), colors):
        d = s["durations"] * unit_ms
        ax.hist(d, bins=bins, alpha=0.6, color=c,
                label=f"{sname} (max {s['stats']['max'] * unit_ms:.3f} ms)")
        ax.axvline(s["stats"]["max"] * unit_ms, color=c, ls="--", lw=1.6)
    ax.axvline(rec, color="red", lw=2.2, label=f"margined estimate ({rec:.3f} ms)")
    ax.set_yscale("log")
    ax.set_xlabel("Execution time (ms)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Execution-time distribution: {name}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
