# Real-Time Systems: Mini Projects (Unit 1)

Python simulations for the Unit 1 lab manual.

| Project | Topic | Folder |
|---|---|---|
| 1.1 | Simulating and measuring task timing (release, response, jitter, WCET, hard/firm/soft utility) | `Mini_Project_1_1_Task_Timing/` |
| 1.2 | Measurement-based WCET estimation and confidence tool | `Mini_Project_1_2_WCET_Tool/` |
| 1.3 | Real-time jitter analyzer under simulated system noise | `Mini_Project_1_3_Jitter_Analyzer/` |

All three scripts share `common/rts_common.py`: a calibrated busy-work job, a
drift-free periodic release loop with optional disturbances, a background
load thread, jitter/statistics helpers, and a PDF table writer.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9 or later. No RTOS or hardware needed — pure software simulation.

## Run

```bash
python Mini_Project_1_1_Task_Timing/mini_project_1_1.py
python Mini_Project_1_2_WCET_Tool/mini_project_1_2.py
python Mini_Project_1_3_Jitter_Analyzer/mini_project_1_3.py
```

Each script writes its charts to a `figures/` folder next to it and its
numeric results to a `results_1_x.pdf` table (both are git-ignored and are re-created on every run). 1.1 takes about a minute; 1.2
a few seconds; 1.3 about 1-2 minutes (it runs 18 periodic experiments).
Exact timing numbers depend on your machine and vary between runs, since
they measure real wall-clock execution time.

## Key results (from one run; will differ on your machine)

**1.1 Task timing** (T = 50 ms, D = 40 ms, 150 jobs)
- Isolated execution time: mean ~16 ms, 95th percentile ~25 ms, measured max ~34 ms — a *measured, not proven* WCET.
- A background busy thread raises release-jitter std from ~0.04 ms to ~3 ms and drops the hit fraction from ~99% to ~90%.
- Adversarial inputs (max work size + deep code path) exceed the ordinary-input maximum, showing random testing alone under-samples the worst case.
- The hard/firm/soft utility curves diverge sharply even though the underlying timing is identical.

**1.2 WCET tool** (4 target functions, 1000 calls per strategy, +54% margin from a 65% utilization budget)
- For functions with data-dependent branches (search, Collatz, nested conditionals), the adversarial strategy finds a maximum 1.3-3x the random strategy's — random testing alone would have missed it.
- For a fixed-control-flow filter, both strategies agree closely; the gap there is timing noise, not a path effect (flagged by the outlier check).
- The margin cannot substitute for path coverage: random-only-times-margin still falls short of the true adversarial maximum in several functions.

**1.3 Jitter analyzer** (T = 20 ms, D = 16 ms, 200 jobs per configuration, 3 disturbances x ~4-5 settings each)
- All three disturbances (background load, random start delay, execution-time variability) increase release and/or completion jitter.
- Several configurations keep the deadline-hit fraction at 95-100% while release jitter grows well beyond the period's normal range — demonstrating that "meets every deadline" and "has low jitter" are independent properties.

## Structure

```
Real_Time_Systems_Mini_Projects/
├── README.md
├── requirements.txt
├── .gitignore
├── common/
│   └── rts_common.py                 shared timing/jitter/statistics/PDF helpers
├── Mini_Project_1_1_Task_Timing/     mini_project_1_1.py
├── Mini_Project_1_2_WCET_Tool/       mini_project_1_2.py, wcet_tool.py
└── Mini_Project_1_3_Jitter_Analyzer/ mini_project_1_3.py
```

The written report is submitted separately and is not part of this repository.
