"""Mini Project 1.1 - Simulating and measuring real-time task timing in Python.

Steps 1-9 of the lab manual plus the four Lab Tasks. Run:

    python mini_project_1_1.py

Outputs (next to this file): figures/*.png and results_1_1.pdf.
Timing numbers vary from machine to machine and run to run.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import rts_common as rc  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------ parameters
T = 0.05                # nominal period (s)
D = 0.8 * T             # relative deadline (s)
N_ISO = 300             # isolated execution-time samples (step 2)
N_JOBS = 150            # jobs in each periodic run (steps 3-4)
SEED = 42
HARD_PENALTY = -10.0    # hard-deadline utility once R > D
TAU = 0.25 * D          # soft-deadline decay constant (s)
MS = 1000.0


def draw_input(rng):
    """Random job input: work size (ms of CPU) and a rare 'deep' code path."""
    work_ms = float(np.clip(rng.normal(14.0, 4.0), 4.0, 30.0))
    deep = bool(rng.random() < 0.10)
    return work_ms, deep


def make_inputs(n, seed):
    rng = np.random.default_rng(seed)
    return [draw_input(rng) for _ in range(n)]


# ------------------------------------------------ step 2: isolated timing
def measure_isolated(inputs):
    d = np.zeros(len(inputs))
    for i, (w, deep) in enumerate(inputs):
        t0 = time.perf_counter()
        rc.numeric_job(w, deep)
        d[i] = time.perf_counter() - t0
    return d


# --------------------------------------- steps 3-6: periodic run + metrics
def periodic_experiment(period, n_jobs, seed, load=None):
    inputs = make_inputs(n_jobs, seed)
    run = rc.run_periodic(lambda k: rc.numeric_job(*inputs[k]), period, n_jobs,
                          0.8 * period, deadline_ref="actual", load=load)
    return run


def utilities(R, d, tau, hard_penalty=HARD_PENALTY):
    hard = np.where(R <= d, 1.0, hard_penalty)
    firm = np.where(R <= d, 1.0, 0.0)
    soft = np.where(R <= d, 1.0, np.exp(-(R - d) / tau))
    return hard, firm, soft


def describe_run(run, tau=TAU):
    rj = rc.release_jitter(run)
    cj = rc.completion_jitter(run)
    R = run["resp"]
    hard, firm, soft = utilities(R, run["D"], tau)
    return dict(
        rel_std=rc.jitter_stats(rj)["std"], rel_max=rc.jitter_stats(rj)["max_abs"],
        comp_std=rc.jitter_stats(cj)["std"], comp_max=rc.jitter_stats(cj)["max_abs"],
        hit=float(run["met"].mean()), R_mean=float(R.mean()), R_max=float(R.max()),
        hard=float(hard.mean()), firm=float(firm.mean()), soft=float(soft.mean()))


# -------------------------------------------------------------- plotting
def plot_timing_diagram(run, path, title):
    n = 12
    miss = np.where(~run["met"])[0]
    start = 0 if miss.size == 0 else int(np.clip(miss[0] - 3, 0, len(run["met"]) - n))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for row, k in enumerate(range(start, start + n)):
        color = "tab:green" if run["met"][k] else "tab:red"
        ax.broken_barh([(run["actual"][k] * MS, run["resp"][k] * MS)],
                       (row - 0.35, 0.7), facecolors=color)
        ax.plot(run["ideal"][k] * MS, row, marker="^", color="k", ms=6)
        ax.plot(run["abs_deadline"][k] * MS, row, marker="|", color="tab:blue",
                ms=16, mew=2.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"job {k}" for k in range(start, start + n)])
    ax.invert_yaxis()
    ax.set_xlabel("Time since start (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="tab:green", label="executing, deadline met"),
        Patch(color="tab:red", label="executing, deadline missed"),
        Line2D([], [], marker="^", color="k", ls="", label="ideal release r0 + kT"),
        Line2D([], [], marker="|", color="tab:blue", ls="", mew=2.5, ms=12,
               label="absolute deadline")], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_resp_util(run, path, title, tau=TAU):
    R = run["resp"]
    hard, firm, soft = utilities(R, run["D"], tau)
    k = np.arange(len(R))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    a1.plot(k, R * MS, "o-", ms=3, lw=1)
    a1.axhline(run["D"] * MS, color="r", ls="--", label=f"D = {run['D'] * MS:.0f} ms")
    a1.set_ylabel("Response time R (ms)")
    a1.set_title(title)
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(k, hard, drawstyle="steps-mid", label="hard", lw=1.4)
    a2.plot(k, firm + 0.02, drawstyle="steps-mid", label="firm", lw=1.4)
    a2.plot(k, soft, drawstyle="steps-mid", label="soft (exp. decay)", lw=1.4)
    a2.set_ylabel("Utility")
    a2.set_xlabel("Job index")
    a2.legend(loc="lower left")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    out = {}
    print(f"Calibration: {rc.ITERS_PER_MS:.0f} loop iterations per ms")

    # ---- step 2: isolated execution time
    iso = measure_isolated(make_inputs(N_ISO, SEED))
    st = rc.summary_stats(iso)
    out["isolated"] = st
    print(f"Isolated C: mean {st['mean'] * MS:.2f} ms, std {st['std'] * MS:.2f} ms, "
          f"max {st['max'] * MS:.2f} ms, p95 {st['p95'] * MS:.2f} ms")

    # ---- steps 3-8: unloaded periodic run
    base = periodic_experiment(T, N_JOBS, SEED)
    out["baseline"] = describe_run(base)
    plot_timing_diagram(base, os.path.join(FIG, "fig1_timing_diagram_unloaded.png"),
                        f"Timing diagram, unloaded (T = {T * MS:.0f} ms, D = {D * MS:.0f} ms)")
    plot_resp_util(base, os.path.join(FIG, "fig2_response_and_utility_unloaded.png"),
                   "Response time and utility, unloaded")

    # ---- Lab Task 1: background busy thread
    loaded = periodic_experiment(T, N_JOBS, SEED, load=rc.LoadThread(1.0))
    out["loaded"] = describe_run(loaded)
    plot_timing_diagram(loaded, os.path.join(FIG, "fig1b_timing_diagram_loaded.png"),
                        "Timing diagram with a background busy thread")
    plot_resp_util(loaded, os.path.join(FIG, "fig2b_response_and_utility_loaded.png"),
                   "Response time and utility with a background busy thread")

    # ---- deadline sweep (uses recorded response times)
    ds = np.linspace(0.4 * base["resp"].mean(), 1.5 * max(base["resp"].max(), 1e-3), 60)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, run in (("unloaded", base), ("background thread", loaded)):
        ax.plot(ds * MS, [(run["resp"] <= d).mean() * 100 for d in ds], label=name)
    for label, val, ls in (("mean R (unloaded)", base["resp"].mean(), ":"),
                           ("95th pct R (unloaded)", np.percentile(base["resp"], 95), "--")):
        ax.axvline(val * MS, color="gray", ls=ls, label=label)
    ax.axvline(D * MS, color="r", alpha=0.6, label="assigned D")
    ax.set_xlabel("Relative deadline D (ms)")
    ax.set_ylabel("Deadline-hit fraction (%)")
    ax.set_title("Hit fraction versus deadline: the tail governs, not the mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_deadline_sweep.png"), dpi=200)
    plt.close(fig)
    out["hit_at_mean_D"] = float((base["resp"] <= base["resp"].mean()).mean())
    out["hit_at_p95_D"] = float((base["resp"] <= np.percentile(base["resp"], 95)).mean())

    # ---- Lab Task 2: adversarial inputs
    rng = np.random.default_rng(SEED + 1)
    adv_inputs = [(float(rng.uniform(26.0, 30.0)), True) for _ in range(150)]
    adv = measure_isolated(adv_inputs)
    out["adversarial"] = rc.summary_stats(adv)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(iso * MS, bins=30, alpha=0.7, label="ordinary inputs")
    ax.hist(adv * MS, bins=30, alpha=0.7, label="adversarial inputs")
    ax.axvline(iso.max() * MS, color="tab:blue", ls="--", label="ordinary max")
    ax.axvline(adv.max() * MS, color="tab:orange", ls="--", label="adversarial max")
    ax.set_xlabel("Execution time (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Ordinary versus adversarial execution times")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_adversarial_vs_ordinary.png"), dpi=200)
    plt.close(fig)

    # ---- Lab Task 3: soft-deadline decay rate
    taus = {"fast decay (0.2 x tau)": 0.2 * TAU, "baseline tau": TAU,
            "slow decay (5 x tau)": 5.0 * TAU}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    late = np.linspace(0, 6 * TAU, 200)
    soft_rows = []
    for name, tau in taus.items():
        a1.plot(late * MS, np.exp(-late / tau), label=name)
        _, _, s = utilities(loaded["resp"], D, tau)
        soft_rows.append((name, tau, float(s.mean())))
    a1.set_xlabel("Lateness R - D (ms)")
    a1.set_ylabel("Soft utility")
    a1.set_title("Soft-deadline utility for three decay rates")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a2.bar([r[0].split(" (")[0] for r in soft_rows], [r[2] for r in soft_rows],
           color=["tab:red", "tab:blue", "tab:green"])
    a2.set_ylabel("Mean soft utility (loaded run)")
    a2.set_ylim(0, 1.05)
    a2.set_title("Same timing, different decay assumption")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_soft_decay.png"), dpi=200)
    plt.close(fig)
    out["soft_rows"] = soft_rows

    # ---- Lab Task 4: vary T with D = 0.8 T
    sweep = []
    for period in (0.03, 0.05, 0.1, 0.2):
        run = periodic_experiment(period, 100, SEED + 2)
        d = describe_run(run, tau=0.25 * 0.8 * period)
        d["T"] = period
        d["rel_std_pct"] = 100 * d["rel_std"] / period
        d["rel_max_pct"] = 100 * d["rel_max"] / period
        sweep.append(d)
    out["period_sweep"] = sweep
    fig, ax = plt.subplots(figsize=(8, 4.4))
    Ts = [s["T"] * MS for s in sweep]
    ax.plot(Ts, [s["rel_std_pct"] for s in sweep], "o-", label="std of release jitter")
    ax.plot(Ts, [s["rel_max_pct"] for s in sweep], "s-", label="max release jitter")
    ax.set_xscale("log")
    ax.set_xlabel("Period T (ms, D = 0.8 T)")
    ax.set_ylabel("Release jitter (% of period)")
    ax.set_title("Jitter as a fraction of the period")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_period_sweep.png"), dpi=200)
    plt.close(fig)

    # ---- PDF results
    f = lambda x: f"{x * MS:.2f}"
    pct = lambda x: f"{100 * x:.1f}%"
    b, l = out["baseline"], out["loaded"]
    sections = [
        dict(heading="1. Isolated execution time (step 2, 300 calls, varying inputs)",
             notes=["The observed maximum is a measured, not proven, WCET estimate: no test "
                    "campaign guarantees the longest path and worst hardware state were exercised "
                    "(Section 1.7). A static analysis gives a sound bound instead."],
             header=["Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "95th pct (ms)"],
             rows=[[f(st["mean"]), f(st["std"]), f(st["min"]), f(st["max"]), f(st["p95"])]]),
        dict(heading=f"2. Periodic runs (T = {T * MS:.0f} ms, D = {D * MS:.0f} ms, {N_JOBS} jobs)",
             notes=["Release jitter = actual - ideal release. Completion jitter = deviation of "
                    "(completion - ideal release) from its mean. Utilities are per-job means."],
             header=["Metric", "Unloaded", "Background thread"],
             rows=[["Mean response R (ms)", f(b["R_mean"]), f(l["R_mean"])],
                   ["Max response R (ms)", f(b["R_max"]), f(l["R_max"])],
                   ["Release jitter std (ms)", f(b["rel_std"]), f(l["rel_std"])],
                   ["Release jitter max |.| (ms)", f(b["rel_max"]), f(l["rel_max"])],
                   ["Completion jitter std (ms)", f(b["comp_std"]), f(l["comp_std"])],
                   ["Completion jitter max |.| (ms)", f(b["comp_max"]), f(l["comp_max"])],
                   ["Deadline-hit fraction", pct(b["hit"]), pct(l["hit"])],
                   ["Mean hard utility (penalty -10)", f"{b['hard']:.2f}", f"{l['hard']:.2f}"],
                   ["Mean firm utility", f"{b['firm']:.2f}", f"{l['firm']:.2f}"],
                   ["Mean soft utility (tau = D/4)", f"{b['soft']:.2f}", f"{l['soft']:.2f}"]]),
        dict(heading="3. Lab Task 2: adversarial versus ordinary inputs",
             notes=["Adversarial inputs: maximum work size with the deep code path (150 calls)."],
             header=["Input set", "Mean (ms)", "95th pct (ms)", "Max (ms)"],
             rows=[["Ordinary (random)", f(st["mean"]), f(st["p95"]), f(st["max"])],
                   ["Adversarial", f(out["adversarial"]["mean"]), f(out["adversarial"]["p95"]),
                    f(out["adversarial"]["max"])]]),
        dict(heading="4. Lab Task 3: soft-deadline decay rate (loaded run)",
             header=["Decay setting", "tau (ms)", "Mean soft utility"],
             rows=[[n, f"{t * MS:.1f}", f"{u:.3f}"] for n, t, u in soft_rows]),
        dict(heading="5. Lab Task 4: varying the period (D = 0.8 T, 100 jobs each)",
             header=["T (ms)", "D (ms)", "Rel. jitter std (ms)", "Rel. jitter max (ms)",
                     "Std (% of T)", "Max (% of T)", "Hit fraction"],
             rows=[[f"{s['T'] * MS:.0f}", f"{0.8 * s['T'] * MS:.0f}", f(s["rel_std"]),
                    f(s["rel_max"]), f"{s['rel_std_pct']:.2f}", f"{s['rel_max_pct']:.2f}",
                    pct(s["hit"])] for s in sweep]),
    ]
    rc.save_tables_pdf(os.path.join(HERE, "results_1_1.pdf"),
                       "Mini Project 1.1: results", sections)
    if os.environ.get("RTS_JSON_OUT"):
        with open(os.environ["RTS_JSON_OUT"], "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    print("Done. Figures in ./figures, tables in results_1_1.pdf")


if __name__ == "__main__":
    main()
