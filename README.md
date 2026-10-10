# TaxMeter

A measurement harness for the running cost of LEO satellite handover predictors.
TaxMeter takes a predictor, a workload and a deployment scenario and reports the
Predictability Tax (PT), a five-axis cost vector of energy per decision, peak memory,
p99.9 latency, dollars per orbit and carbon per gigabyte delivered, each read against
the scenario's budget, together with the worst-case budget fraction and the axis that
binds. This repository is the clean-room rebuild of the harness from its written
specification (implementation B in the paper). Everything the written specification
does not determine is logged in SPEC_GAPS.md.

## Repository layout

```
taxmeter/core.py               Walker-delta constellation (SGP4), geometry, Ku-band link
                               budget, event workload generation
taxmeter/predictors.py         nine predictors with pure-NumPy inference (eight from the
                               paper plus a vectorized Kalman variant) and the batched
                               featurizers shared with training
taxmeter/train_torch.py        supervised training of the four learned predictors (CPU
                               torch); weights are exported into the NumPy layout
taxmeter/measure.py            the measurement protocol, bootstrap intervals on p99.9,
                               PT axes, budget sweep, TDP and RAPL energy backends
taxmeter/cli.py                taxmeter measure --predictor X --scenario Y --output Z
run_all.py                     end to end: workload, training, evaluation, measurement
experiments/export_paper_tables.py   writes results/paper_table1.csv (A quoted, B measured)
experiments/op_counts_and_scaling.py MAC counts per decision and cost against candidate count
experiments/mac_energy.py      measured energy per decision on Apple Silicon (powermetrics)
experiments/meter_energy.py    measured energy per decision with an external USB or USB-C power meter
experiments/batching.py        per-decision cost of every predictor at batch sizes 1 to 512
experiments/cascade.py         a cost-aware cascade (geometric rule, escalating close calls to LSTM-Attn)
survey/                        the surveyed handover papers and what each reports
tests/test_taxmeter.py         smoke tests (parameter counts, workload shapes, protocol)
data/workload.npz              the 7,205-event synthetic-shell workload (implementation B)
data/starlink_celestrak.tle    CelesTrak Starlink TLE snapshot of 3 October 2026 (10,679 satellites)
data/workload_live.npz         the 4,800-event live-constellation workload on Starlink's 15 s slots (5 cities, 4 h)
data/workload_live20.npz       the 28,800-slot live workload over twenty cities, 34 S to 64 N, 6 h
results/live20_per_city.csv    per-city accuracy of all predictors and mean visible satellites
weights_live/*.npz             weights trained on the live-constellation workload
weights/*.npz                  trained weights for MLP, DRL, LSTM-Attn and GNN
results/                       CSVs behind the paper's table and figures; PROVENANCE.md maps
                               every number in the paper to one file
SPEC_GAPS.md                   the eleven under-specified points and what the rebuild chose
AI_USE.md                      how generative AI was used to build this harness and run the experiments
equipment.txt                  hosts, versions and isolation for every number here
requirements.txt               pinned versions
```

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install numpy==2.4.4 sgp4==2.27
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # training only
```

## Reproduce the paper's implementation-B numbers

```
python run_all.py --skip-workload --skip-train      # uses data/ and weights/ as shipped
python experiments/export_paper_tables.py           # results/paper_table1.csv
python tests/test_taxmeter.py                        # five smoke tests
```

The full pipeline (`python run_all.py`) regenerates the workload from the FCC 21-48
Gen-1 53 degree shell with SGP4 and retrains the four learned predictors; it takes about
two minutes on one core and is deterministic for a fixed seed on a fixed host.
`--full-gen1` builds all 4,408 licensed satellites. The live-constellation run of the paper is

```
python run_all.py --tle data/starlink_celestrak.tle --starlink-slots \
    --workload data/workload_live.npz --weights weights_live --tag _live
```

which propagates the CelesTrak snapshot with SGP4 and places one decision on each of
Starlink's reconfiguration slots (the 12th, 27th, 42nd and 57th second of every minute).
The twenty-city run uses `core.generate_workload(..., cities=core.CITIES_20, hours=6.0)`
and evaluates the weights trained on the five-city live run; the per-city table is
`results/live20_per_city.csv`.

## Timing grades

Latency and TDP-proxy energy depend on the host. `run_all.py` labels every run:

- `devgrade` (default): any host; this is what the shipped results are.
- `canonical`: `run_all.py --canonical` refuses to run unless the process is pinned to
  one core and every CPU reports the `performance` frequency governor. Use
  `taskset -c <core>` and `cpupower frequency-set -g performance` on an idle machine.

`measure.RAPLBackend` reads Intel RAPL package counters where `/sys/class/powercap` is
present; it is implemented and untested on the development host. Pair it with an
external meter.

## Measured energy on an Apple Silicon Mac

`experiments/mac_energy.py` measures energy per decision with Apple's `powermetrics`, which
reports the CPU power from the chip's own telemetry. Each predictor runs single-threaded in
a tight loop while powermetrics samples every 100 ms; an idle baseline is sampled with the
process asleep; energy per decision is the marginal power divided by decisions per second.
Repetitions are interleaved across predictors.

```
sudo -v                                   # powermetrics needs administrator rights
caffeinate -i python experiments/mac_energy.py
```

Outputs: `results/mac_energy_runs.csv`, `results/mac_energy_summary.csv` and the raw
powermetrics text in `results/mac_energy_powermetrics/`. Run on mains power with other
applications closed.

## Measured energy with an external USB or USB-C power meter

`experiments/meter_energy.py` measures energy per decision with a power meter placed between
the power supply and the machine: a USB meter in front of a Raspberry Pi, or a USB-C meter
in front of a MacBook. It asks for the meter's accumulated-energy counter (mWh) at the start
and end of every window and keeps the predictor running until the second reading is typed,
so the reading delay is inside the window at both ends.

```
python experiments/meter_energy.py                       # 2 repetitions x 120 s
python experiments/meter_energy.py --seconds 10 --reps 1 # rehearsal
```

Step-by-step guides for both machines come with the measurement kit (README_MAC.md and
README_RASPBERRY_PI.md).

## What the shipped results are and are not

The energy axis is wall-clock time multiplied by the scenario TDP (15 W satellite,
150 W ground). It is a proxy, used by the paper only for orderings that span orders of
magnitude. The workload is synthetic (SGP4 geometry, a Ku-band link budget with
shadowing and scintillation, one event per 10 s). Neither is a measurement of a live
Starlink link; see SPEC_GAPS.md for every modelling choice.

## Citation

Please cite the HotMobile 2027 paper "The Predictability Tax of LEO Handover
Predictors" once it is published; the artifact is released under the MIT License.
