"""Shared helpers for the Real-Time Systems mini projects (Unit 1).

Provides: calibrated busy-work jobs, a drift-free periodic release loop with
optional disturbances, a duty-cycle background load thread, jitter/statistics
helpers and a small routine that writes result tables to PDF.

This is a software simulation for illustrating timing concepts. It does not
produce certifiable timing guarantees.
"""
import math
import threading
import time

import numpy as np


# ---------------------------------------------------------------- busy work
def _spin(n):
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)
    return s


def calibrate_iters_per_ms(repeats=5):
    """Return how many _spin iterations take about one millisecond here."""
    _spin(20000)                                   # warm-up
    best = 0.0
    n = 50000
    for _ in range(repeats):
        t0 = time.perf_counter()
        _spin(n)
        dt = time.perf_counter() - t0
        best = max(best, n / (dt * 1000.0))
    return best


ITERS_PER_MS = calibrate_iters_per_ms()


def busy_ms(ms):
    """Burn roughly `ms` milliseconds of CPU (calibrated, not exact)."""
    return _spin(max(1, int(ms * ITERS_PER_MS)))


def numeric_job(work_ms, deep=False):
    """Job body for Mini Projects 1.1/1.3: work grows with `work_ms`; the
    `deep` flag takes a longer code path (an extra half-size pass)."""
    s = _spin(max(1, int(work_ms * ITERS_PER_MS)))
    if deep:
        s += _spin(max(1, int(0.5 * work_ms * ITERS_PER_MS)))
    return s


# ------------------------------------------------------- background thread
class LoadThread(threading.Thread):
    """Background CPU load. intensity in [0, 1] is the duty cycle over a
    10 ms cycle (1.0 = continuous busy work)."""

    def __init__(self, intensity=1.0, cycle_s=0.010):
        super().__init__(daemon=True)
        self.intensity = intensity
        self.cycle_s = cycle_s
        self._stop_evt = threading.Event()

    def run(self):
        x = 1.0
        while not self._stop_evt.is_set():
            if self.intensity <= 0:
                time.sleep(self.cycle_s)
                continue
            end = time.perf_counter() + self.cycle_s * self.intensity
            while time.perf_counter() < end:
                x = x * 1.0000001 + 1e-9
            rest = self.cycle_s * (1.0 - self.intensity)
            if rest > 0:
                time.sleep(rest)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)


# ------------------------------------------------------------ periodic loop
def run_periodic(job, T, n_jobs, D, deadline_ref="actual",
                 stall_prob=0.0, stall_s=0.0, rng=None, load=None):
    """Run `job(k)` periodically and record timing instants.

    * Release k is ideally at r0 + k*T. The loop sleeps until that absolute
      target (recomputed from r0 each iteration, so there is no drift).
    * stall_prob / stall_s: with this probability an extra stall delays the
      job's start (interrupt / DMA-like disruption).
    * The recorded 'actual release' is the instant the job is dispatched.
    * deadline_ref: 'actual' -> absolute deadline = actual release + D
      (Mini Project 1.1); 'ideal' -> ideal release + D (Mini Project 1.3).
    * load: optional LoadThread started for the duration of the run.
    Returns a dict of numpy arrays (seconds, relative to r0).
    """
    rng = rng or np.random.default_rng(0)
    if load is not None:
        load.start()
        time.sleep(0.05)
    ideal = np.zeros(n_jobs)
    actual = np.zeros(n_jobs)
    done = np.zeros(n_jobs)
    r0 = time.perf_counter() + 0.05
    try:
        for k in range(n_jobs):
            target = r0 + k * T
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            if stall_prob > 0 and rng.random() < stall_prob:
                time.sleep(stall_s)
            actual[k] = time.perf_counter() - r0
            ideal[k] = k * T
            job(k)
            done[k] = time.perf_counter() - r0
    finally:
        if load is not None:
            load.stop()
    resp = done - actual
    if deadline_ref == "actual":
        abs_deadline = actual + D
    else:
        abs_deadline = ideal + D
    met = done <= abs_deadline + 1e-12
    return dict(ideal=ideal, actual=actual, done=done, resp=resp,
                abs_deadline=abs_deadline, met=met, T=T, D=D)


# ---------------------------------------------------------------- analysis
def release_jitter(run):
    return run["actual"] - run["ideal"]


def completion_jitter(run):
    """Deviation of each completion from the mean latency measured against
    the ideal periodic grid: (c_k - ideal_k) - mean(c - ideal)."""
    lat = run["done"] - run["ideal"]
    return lat - lat.mean()


def jitter_stats(x):
    return dict(std=float(np.std(x)), max_abs=float(np.max(np.abs(x))))


def summary_stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)),
                min=float(x.min()), max=float(x.max()),
                p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)))


# --------------------------------------------------------------- PDF tables
def save_tables_pdf(path, title, sections, page_size=(8.27, 11.69)):
    """Write result tables to a PDF. `sections` is a list of dicts with keys
    heading, header (list), rows (list of lists), and optional notes (list of
    str). Each section gets its own page."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for sec in sections:
            fig = plt.figure(figsize=page_size)
            fig.text(0.06, 0.965, title, fontsize=13, weight="bold", va="top")
            fig.text(0.06, 0.935, sec["heading"], fontsize=11, weight="bold", va="top")
            y = 0.905
            for note in sec.get("notes", []):
                wrapped = textwrap.fill(note, 105)
                fig.text(0.06, y, wrapped, fontsize=8, va="top")
                y -= 0.016 * (wrapped.count("\n") + 1) + 0.008
            n_rows = len(sec["rows"]) + 1
            height = min(0.05 * n_rows, y - 0.06)
            ax = fig.add_axes([0.04, y - height - 0.01, 0.92, height])
            ax.axis("off")
            tab = ax.table(cellText=sec["rows"], colLabels=sec["header"],
                           loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(7.5)
            tab.auto_set_column_width(list(range(len(sec["header"]))))
            tab.scale(1, 1.6)
            for (r, c), cell in tab.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#D9E2F3")
                    cell.set_text_props(weight="bold")
            pdf.savefig(fig)
            plt.close(fig)
