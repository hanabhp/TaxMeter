#!/usr/bin/env python3
"""Measured energy per decision on an Apple Silicon Mac, using powermetrics.

What it does
------------
For every predictor it runs the forward pass in a tight single-threaded loop while
Apple's powermetrics samples the CPU power reported by the chip every 100 ms. It
also samples an idle baseline with the process asleep. Energy per decision is

    (mean CPU power while looping - mean idle CPU power) / decisions per second

The same loop gives the mean time per decision, from which the script also computes
the paper's proxy (15 W x time) on this Mac, so measured and modelled energy can be
compared predictor by predictor. Repetitions are interleaved across predictors so
slow thermal drift does not favour any one of them.

Usage (from the artifact root)
------------------------------
    sudo -v                                # type your Mac password once
    python3 experiments/mac_energy.py      # about 12 minutes with the defaults

Outputs
-------
    results/mac_energy_runs.csv      one row per predictor per repetition
    results/mac_energy_summary.csv   one row per predictor (mean and spread)
    results/mac_energy_powermetrics/ raw powermetrics text, kept for audit
"""
import os
# Single-threaded numerics, matching the paper's protocol. Must be set before numpy loads.
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from taxmeter.predictors import build_events, load_predictors  # noqa: E402

PROXY_W = 15.0          # the paper's satellite-scenario TDP
RE_ELAPSED = re.compile(r"\(([\d.]+)ms elapsed\)")
RE_CPU = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.M)


def parse_powermetrics(text):
    """Return (mean CPU power in W, number of samples), weighting by sample length.
    The first sample is dropped because it overlaps process start-up."""
    blocks = text.split("*** Sampled system activity")[1:]
    pw, dt = [], []
    for b in blocks:
        e, c = RE_ELAPSED.search(b), RE_CPU.search(b)
        if e and c:
            dt.append(float(e.group(1)))
            pw.append(float(c.group(1)) / 1000.0)
    pw, dt = pw[1:], dt[1:]
    if not pw:
        raise RuntimeError("no CPU Power samples parsed; is this an Apple Silicon Mac?")
    return sum(p * d for p, d in zip(pw, dt)) / sum(dt), len(pw)


def run_window(seconds, interval_ms, out_path, work=None):
    """Sample powermetrics for `seconds` while `work` (a callable doing one decision)
    runs in a loop; with work=None the process sleeps (idle baseline)."""
    n = max(3, int(seconds * 1000 / interval_ms))
    subprocess.run(["sudo", "-n", "-v"], capture_output=True)   # keep the sudo timestamp fresh
    with open(out_path, "w") as f:
        p = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
                              "-i", str(interval_ms), "-n", str(n)],
                             stdout=f, stderr=subprocess.STDOUT)
        calls, t0 = 0, time.perf_counter()
        if work is None:
            p.wait()
        else:
            while p.poll() is None:
                for _ in range(200):
                    work()
                calls += 200
        wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"powermetrics failed, see {out_path}")
    watts, samples = parse_powermetrics(open(out_path).read())
    return watts, samples, calls, wall


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=15.0, help="measurement window per predictor and repetition")
    ap.add_argument("--reps", type=int, default=3, help="interleaved repetitions")
    ap.add_argument("--interval-ms", type=int, default=100, help="powermetrics sampling interval")
    ap.add_argument("--events", type=int, default=300, help="events cycled through by each predictor")
    args = ap.parse_args()

    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit("This script needs an Apple Silicon Mac (macOS on arm64).")
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        sys.exit("Run `sudo -v` first and type your password, then rerun this script.")

    raw_dir = os.path.join(ROOT, "results", "mac_energy_powermetrics")
    os.makedirs(raw_dir, exist_ok=True)
    d = dict(np.load(os.path.join(ROOT, "data", "workload.npz")))
    events, _ = build_events(d)
    rng = np.random.default_rng(7)
    sample = [events[i] for i in rng.choice(len(events), size=min(args.events, len(events)), replace=False)]
    preds = load_predictors(os.path.join(ROOT, "weights"))
    print(f"[mac_energy] {platform.platform()} | {len(preds)} predictors | "
          f"{args.reps} reps x {args.seconds:.0f} s | about "
          f"{(len(preds) + 1) * args.reps * (args.seconds + 1) / 60:.0f} min")

    rows = []
    for rep in range(args.reps):
        idle_w, _, _, _ = run_window(args.seconds, args.interval_ms,
                                     os.path.join(raw_dir, f"idle_rep{rep}.txt"))
        print(f"[rep {rep + 1}/{args.reps}] idle CPU power {idle_w:.3f} W")
        order = preds if rep % 2 == 0 else list(reversed(preds))
        for p in order:
            state = {"i": 0}

            def one(p=p, state=state):
                ev = sample[state["i"]]
                state["i"] = (state["i"] + 1) % len(sample)
                p.predict(ev)
            for _ in range(2000):            # warm-up outside the window
                one()
            w, ns, calls, wall = run_window(args.seconds, args.interval_ms,
                                            os.path.join(raw_dir, f"{p.name}_rep{rep}.txt"), one)
            rate = calls / wall
            marg = max(w - idle_w, 0.0)
            e_uj = marg / rate * 1e6
            proxy_uj = PROXY_W / rate * 1e6
            rows.append(dict(rep=rep, predictor=p.name, idle_w=round(idle_w, 4), active_w=round(w, 4),
                             marginal_w=round(marg, 4), calls=calls, wall_s=round(wall, 3),
                             mean_us=round(1e6 / rate, 3), measured_uj=round(e_uj, 3),
                             proxy_15w_uj=round(proxy_uj, 3), samples=ns))
            print(f"  {p.name:14s} {1e6 / rate:9.2f} us/decision  active {w:6.3f} W  "
                  f"marginal {marg:6.3f} W  measured {e_uj:9.2f} uJ  proxy {proxy_uj:9.2f} uJ")

    out = os.path.join(ROOT, "results", "mac_energy_runs.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    summ = []
    for p in preds:
        r = [x for x in rows if x["predictor"] == p.name]
        m = [x["measured_uj"] for x in r]
        summ.append(dict(predictor=p.name, measured_uj_mean=round(statistics.mean(m), 3),
                         measured_uj_sd=round(statistics.stdev(m), 3) if len(m) > 1 else 0.0,
                         marginal_w_mean=round(statistics.mean(x["marginal_w"] for x in r), 4),
                         mean_us=round(statistics.mean(x["mean_us"] for x in r), 3),
                         proxy_15w_uj=round(statistics.mean(x["proxy_15w_uj"] for x in r), 3)))
    for s in summ:
        s["measured_over_proxy"] = round(s["measured_uj_mean"] / s["proxy_15w_uj"], 4)
    out2 = os.path.join(ROOT, "results", "mac_energy_summary.csv")
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ[0].keys()); w.writeheader(); w.writerows(summ)

    # how the measurement compares with the paper's implementation-B energies
    paper = {}
    pt = os.path.join(ROOT, "results", "paper_table1.csv")
    if os.path.exists(pt):
        for r in csv.DictReader(open(pt)):
            paper[r["predictor"]] = float(r["E_uj_B"])
    common = [s for s in summ if s["predictor"] in paper]
    meas = [s["measured_uj_mean"] for s in summ]
    weak = [s["predictor"] for s in summ if s["marginal_w_mean"] < 0.05]
    if weak:
        print(f"\n[warning] marginal power under 0.05 W for {', '.join(weak)}; quit other apps, "
              f"keep the charger in, and rerun with --seconds 30")
    print("\n[summary]")
    print(f"  measured energy spread across predictors: {max(meas) / max(min(meas), 1e-9):.0f}x")
    print(f"  measured / 15 W proxy on this Mac: {min(s['measured_over_proxy'] for s in summ):.3f} "
          f"to {max(s['measured_over_proxy'] for s in summ):.3f}")
    print(f"  rank correlation, measured vs proxy on this Mac: "
          f"{spearman(meas, [s['proxy_15w_uj'] for s in summ]):.2f}")
    if common:
        print(f"  rank correlation, measured vs paper column B ({len(common)} predictors): "
              f"{spearman([s['measured_uj_mean'] for s in common], [paper[s['predictor']] for s in common]):.2f}")
    print(f"  wrote {out}\n  wrote {out2}")


if __name__ == "__main__":
    main()
