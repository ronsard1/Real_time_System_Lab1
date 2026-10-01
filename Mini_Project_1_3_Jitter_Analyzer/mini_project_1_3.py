"""Mini Project 1.3 - Real-time jitter analyzer under simulated system noise.

Baseline periodic loop plus three independently configurable disturbances:
  A. background CPU load thread (duty-cycle intensity)      -> general interference / OS jitter
  B. random extra start delay (probability, magnitude)      -> interrupt / DMA-stall-like disruption
  C. execution-time variability (random distribution)       -> cache / pipeline-like variation

    python mini_project_1_3.py

Outputs: figures/*.png and results_1_3.pdf. Takes roughly 1.5-2 minutes.
Timing numbers vary from machine to machine and run to run; on a multi-core
machine the background-load effect is usually smaller than on a single core.
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
import rts_common as rc  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------ parameters
T = 0.020               # period (s)
D = 0.8 * T             # fixed deadline after each IDEAL release (16 ms)
N_JOBS = 200            # jobs per experiment
BASE_MS = 4.0           # baseline job body (ms of CPU)
SEED = 7
HIT_MIN = 0.95          # "still meets (almost) every deadline"
JITTER_FRAC = 0.05      # "substantial jitter": release-jitter std >= 5% of the period
MS = 1000.0


# -------------------------------------------------- workload distributions
def workload_sampler(kind, spread, n, rng, median_ms=BASE_MS):
    """Per-job work sizes (ms). kind: 'fixed', 'normal' (truncated) or
    'lognormal' (heavy-tailed). `spread` is sigma (ms for normal, log-space
    sigma for lognormal)."""
    if kind == "fixed" or spread == 0:
        w = np.full(n, median_ms)
    elif kind == "normal":
        w = rng.normal(median_ms, spread, n)
    elif kind == "lognormal":
        w = median_ms * rng.lognormal(0.0, spread, n)
    else:
        raise ValueError(kind)
    return np.clip(w, 0.5, 60.0)


# ------------------------------------------------------------- one experiment
def experiment(load=0.0, stall_prob=0.0, stall_ms=0.0, kind="fixed", spread=0.0,
               seed=SEED):
    rng = np.random.default_rng(seed)
    work = workload_sampler(kind, spread, N_JOBS, rng)
    thread = rc.LoadThread(load) if load > 0 else None
    run = rc.run_periodic(lambda k: rc.busy_ms(work[k]), T, N_JOBS, D,
                          deadline_ref="ideal", stall_prob=stall_prob,
                          stall_s=stall_ms / MS, rng=np.random.default_rng(seed + 100),
                          load=thread)
    rj, cj = rc.release_jitter(run), rc.completion_jitter(run)
    return dict(rel_mean=float(rj.mean()), rel_std=rc.jitter_stats(rj)["std"], rel_max=rc.jitter_stats(rj)["max_abs"],
                comp_std=rc.jitter_stats(cj)["std"], comp_max=rc.jitter_stats(cj)["max_abs"],
                hit=float(run["met"].mean()), R_mean=float(run["resp"].mean()),
                rj_series=rj.tolist())


def sweep_plot(ax, xs, rows, xlabel, title):
    """Release jitter (std, max) versus the disturbance parameter, with the
    deadline-hit fraction on a second axis."""
    ax.plot(xs, [r["rel_std"] * MS for r in rows], "o-", label="release jitter std")
    ax.plot(xs, [r["rel_max"] * MS for r in rows], "s-", label="release jitter max")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Release jitter (ms)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, [r["hit"] * 100 for r in rows], "^--", color="tab:green",
             label="deadline-hit fraction")
    ax2.set_ylabel("Hit fraction (%)", color="tab:green")
    ax2.set_ylim(-3, 105)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")


def main():
    all_rows = []      # (mechanism, setting label, dict)

    def record(mech, label, res):
        all_rows.append((mech, label, res))
        print(f"{mech:<22}{label:<22} rel std {res['rel_std'] * MS:6.2f} ms  "
              f"max {res['rel_max'] * MS:6.2f} ms | hit {res['hit'] * 100:5.1f}%")

    # ---- control condition
    base = experiment()
    record("baseline (control)", "no disturbance", base)

    # ---- A: background CPU load
    loads = [0.25, 0.5, 0.75, 1.0]
    load_rows = [base]
    for x in loads:
        r = experiment(load=x)
        load_rows.append(r)
        record("A: background load", f"duty {x:.2f}", r)

    # ---- B: random extra delay (probability sweep, magnitude fixed 5 ms)
    probs = [0.02, 0.05, 0.10, 0.20]
    prob_rows = []
    for p_ in probs:
        r = experiment(stall_prob=p_, stall_ms=5.0)
        prob_rows.append(r)
        record("B: delay probability", f"p={p_:.2f}, 5 ms", r)
    # ---- B: magnitude sweep (probability fixed 0.10)
    mags = [2.0, 5.0, 8.0, 12.0]
    mag_rows = []
    for m in mags:
        r = experiment(stall_prob=0.10, stall_ms=m)
        mag_rows.append(r)
        record("B: delay magnitude", f"p=0.10, {m:.0f} ms", r)

    # ---- C: execution-time variability (lognormal, median = baseline work)
    sigmas = [0.0, 0.5, 0.8, 1.0, 1.2]
    var_rows = []
    for s in sigmas:
        r = experiment(kind="lognormal", spread=s)
        var_rows.append(r)
        if s == 0.0:      # sigma = 0 is a second control run (no disturbance)
            record("baseline (control)", "control repeat (sigma=0)", r)
        else:
            record("C: exec-time variability", f"lognormal sigma={s:.1f}", r)

    # ---- sweep plots (one per disturbance mechanism)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sweep_plot(ax, [0.0] + loads, load_rows, "Background-load duty cycle",
               "A: release jitter vs background CPU load")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_load_sweep.png"), dpi=200); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    sweep_plot(a1, probs, prob_rows, "Stall probability (magnitude 5 ms)",
               "B1: release jitter vs delay probability")
    sweep_plot(a2, mags, mag_rows, "Stall magnitude (ms, probability 0.10)",
               "B2: release jitter vs delay magnitude")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_delay_sweeps.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sweep_plot(ax, sigmas, var_rows, "Lognormal sigma (spread of job work)",
               "C: release jitter vs execution-time variability")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_variability_sweep.png"), dpi=200); plt.close(fig)

    # ---- combined view: jitter versus hit fraction
    jitter_thr = JITTER_FRAC * T          # 1 ms for T = 20 ms
    flagged = [(m, l, r) for m, l, r in all_rows
               if m != "baseline (control)" and r["hit"] >= HIT_MIN
               and r["rel_std"] >= jitter_thr]
    colors = {"baseline (control)": "k", "A: background load": "tab:red",
              "B: delay probability": "tab:blue", "B: delay magnitude": "tab:cyan",
              "C: exec-time variability": "tab:purple"}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhspan(HIT_MIN * 100, 101, color="tab:green", alpha=0.10)
    ax.axvline(jitter_thr * MS, color="gray", ls=":")
    for mech in colors:
        pts = [r for m, l, r in all_rows if m == mech]
        ax.scatter([r["rel_std"] * MS for r in pts], [r["hit"] * 100 for r in pts],
                   s=55, color=colors[mech], label=mech, zorder=3)
    ax.text(jitter_thr * MS * 1.05, 0.04,
            f"{JITTER_FRAC * 100:.0f}% of T ({jitter_thr * MS:.0f} ms)", fontsize=8,
            color="gray", transform=ax.get_xaxis_transform())
    ax.text(0.01, HIT_MIN * 100 + 0.3, f"hit fraction >= {HIT_MIN * 100:.0f}%",
            fontsize=8, color="tab:green", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("Release jitter std (ms, log scale)")
    ax.set_ylabel("Deadline-hit fraction (%)")
    ax.set_title("Jitter can grow a lot while (almost) every deadline is still met")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_jitter_vs_hit_fraction.png"), dpi=200); plt.close(fig)

    # ---- example: highest-jitter configuration with a (near-)perfect hit rate
    example = None
    if flagged:
        best_hit = max(r["hit"] for _, _, r in flagged)
        cand = [(m, l, r) for m, l, r in flagged if r["hit"] == best_hit]
        example = max(cand, key=lambda t: t[2]["rel_std"])
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(np.array(base["rj_series"]) * MS, lw=1, label="baseline")
        ax.plot(np.array(example[2]["rj_series"]) * MS, lw=1,
                label=f"{example[0]}, {example[1]} (hit {example[2]['hit'] * 100:.0f}%)")
        ax.set_xlabel("Job index")
        ax.set_ylabel("Release jitter (ms)")
        ax.set_title("Same hit fraction, very different regularity")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_example_low_vs_high_jitter.png"), dpi=200); plt.close(fig)

    # ---- PDF results
    ms = lambda v: f"{v * MS:.2f}"
    rows = [[m, l, ms(r["rel_mean"]), ms(r["rel_std"]), ms(r["rel_max"]), ms(r["comp_std"]),
             ms(r["comp_max"]), f"{r['hit'] * 100:.1f}%", ms(r["R_mean"])] for m, l, r in all_rows]
    frows = [[m, l, ms(r["rel_std"]), ms(r["rel_max"]), f"{100 * r['rel_std'] / T:.1f}%",
              f"{r['hit'] * 100:.1f}%"] for m, l, r in flagged]
    sections = [
        dict(heading="1. All tested settings",
             notes=[f"T = {T * MS:.0f} ms, deadline D = {D * MS:.0f} ms after each ideal release, "
                    f"{N_JOBS} jobs per experiment, baseline job body {BASE_MS:.0f} ms of CPU. "
                    "Release jitter = actual - ideal release. Completion jitter = deviation of "
                    "(completion - ideal release) from its mean. Times in ms."],
             header=["Mechanism", "Setting", "Rel. mean", "Rel. std", "Rel. max", "Comp. std",
                     "Comp. max", "Hit fraction", "Mean R"],
             rows=rows),
        dict(heading="2. Jitter grew, hit fraction stayed high",
             notes=[f"Configurations with hit fraction >= {HIT_MIN * 100:.0f}% and release-jitter std "
                    f">= {JITTER_FRAC * 100:.0f}% of the period ({jitter_thr * MS:.0f} ms). These show that "
                    "meeting deadlines and having low jitter are distinct properties."],
             header=["Mechanism", "Setting", "Rel. std (ms)", "Rel. max (ms)", "Std as % of T",
                     "Hit fraction"],
             rows=frows or [["(none)", "", "", "", "", ""]]),
    ]
    rc.save_tables_pdf(os.path.join(HERE, "results_1_3.pdf"),
                       "Mini Project 1.3: results", sections)
    if os.environ.get("RTS_JSON_OUT"):
        dump = [dict(mech=m, label=l, **{k: v for k, v in r.items() if k != "rj_series"})
                for m, l, r in all_rows]
        with open(os.environ["RTS_JSON_OUT"], "w") as fh:
            json.dump(dict(rows=dump, flagged=[[m, l] for m, l, _ in flagged],
                           example=[example[0], example[1]] if example else None),
                      fh, indent=1, default=float)
    print("Done. Figures in ./figures, tables in results_1_3.pdf")


if __name__ == "__main__":
    main()
