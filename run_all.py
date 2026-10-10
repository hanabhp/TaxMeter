#!/usr/bin/env python3
"""run_all.py — end-to-end TaxMeter rebuild.

Stages: workload -> train -> measure.  Timing on a shared dev box is
DEVELOPMENT-GRADE; canonical numbers come from an idle bare-metal machine
(see README.md, RAPL backend).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxmeter import core, measure
from taxmeter.predictors import (build_events, featurize_all, load_predictors,
                                 param_report, STATED_PARAMS)

def host_is_canonical():
    """True only when pinned to a single core with the performance governor."""
    import glob
    try:
        pinned = len(os.sched_getaffinity(0)) == 1
    except AttributeError:
        pinned = False
    govs = []
    for f in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            govs.append(open(f).read().strip())
        except OSError:
            pass
    performance = bool(govs) and all(g == "performance" for g in govs)
    return pinned and performance, pinned, performance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-gen1", action="store_true",
                    help="use all 4,408 Gen-1 satellites (default: the "
                         "1,584-sat 53.0 deg shell the paper's text describes)")
    ap.add_argument("--skip-workload", action="store_true")
    ap.add_argument("--tle", default=None,
                    help="live TLE file (e.g. data/starlink_celestrak.tle) instead of Walker-delta")
    ap.add_argument("--starlink-slots", action="store_true",
                    help="decide on Starlink's 15 s slots at :12/:27/:42/:57 instead of every 10 s")
    ap.add_argument("--workload", default="data/workload.npz")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--canonical", action="store_true",
                    help="refuse to run unless pinned to one core with the "
                         "performance governor; results are then labelled canonical")
    args = ap.parse_args()

    ok, pinned, perf = host_is_canonical()
    if args.canonical and not ok:
        sys.exit(f"[canonical] refused: pinned={pinned} performance_governor={perf}. "
                 "Run under `taskset -c <core>` with `cpupower frequency-set -g performance`.")
    grade = "canonical" if (args.canonical and ok) else "devgrade"
    print(f"[host] timing grade: {grade} (pinned={pinned}, performance_governor={perf})")

    wl = args.workload
    if not args.skip_workload or not os.path.exists(wl):
        shells = core.SHELLS_GEN1_FULL if args.full_gen1 else core.SHELLS_SHELL1
        t0 = time.time()
        kw = dict(tick_s=15, tick_offset=12) if args.starlink_slots else {}
        if args.tle:
            kw["tle_path"] = args.tle; kw["epoch"] = (2026, 10, 3)
        core.generate_workload(shells=shells, seed=args.seed, out=wl, **kw)
        print(f"[workload] {time.time()-t0:.1f} s")

    d = dict(np.load(wl))
    n = len(d["label"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_te = int(round(0.2 * n))
    test_idx, train_idx = perm[:n_te], perm[n_te:]
    print(f"[split] {n} events -> train {len(train_idx)} / test {len(test_idx)}")

    feats = featurize_all(d)
    if not args.skip_train:
        from taxmeter.train_torch import train_all
        t0 = time.time()
        train_all(feats, d["label"], train_idx, test_idx, args.weights,
                  seed=args.seed)
        print(f"[train] {time.time()-t0:.1f} s")

    events, _ = build_events(d)
    preds = load_predictors(args.weights)

    print("\n[fidelity] parameter counts (stated vs rebuilt):")
    for name, stated, rebuilt in param_report(preds):
        delta = "" if not stated else f"  ({100*(rebuilt-stated)/stated:+.1f}%)"
        print(f"  {name:14s} stated {stated:6d}  rebuilt {rebuilt:6d}{delta}")

    rows = []
    for p in preds:
        acc, f1 = measure.evaluate(p, events, test_idx)
        t0 = time.time()
        cost = measure.measure_cost(p, events)
        sat = measure.pt_axes(cost, "satellite")
        gnd = measure.pt_axes(cost, "ground")
        rows.append(dict(
            predictor=p.name,
            params_stated=STATED_PARAMS.get(p.name, 0),
            params_rebuilt=p.params,
            acc=round(acc, 3), macro_f1=round(f1, 3),
            mean_us=round(cost["mean_s"] * 1e6, 1),
            p999_ms_devbox=round(cost["p999_ms"], 3),
            mem_mb=round(cost["mem_mb"], 3),
            E_uj_sat=round(sat["E_uj"], 1),
            E_uj_gnd=round(gnd["E_uj"], 1),
            D_orbit_usd=f'{sat["D_orbit"]:.2e}',
            C_g_gb_sat=f'{sat["C_g_gb"]:.2e}',
            PT_star_sat=round(sat["PT_star"], 2),
            bind_sat=sat["binding"],
            PT_star_gnd=round(gnd["PT_star"], 2),
            bind_gnd=gnd["binding"],
        ))
        print(f"[measure] {p.name:14s} acc {acc:.3f}  f1 {f1:.3f}  "
              f"mean {cost['mean_s']*1e6:7.1f} us  p99.9 {cost['p999_ms']:7.3f} ms  "
              f"PT*sat {sat['PT_star']:8.2f}%  ({time.time()-t0:.1f} s)")

    measure.write_csv(f"results/table1_{grade}{args.tag}.csv", rows)
    measure.write_csv(f"results/e1_budget_sweep_{grade}{args.tag}.csv", measure.budget_sweep(rows))

    e = [r["E_uj_sat"] for r in rows]
    l = [r["p999_ms_devbox"] for r in rows]
    print(f"\n[headline] energy spread {max(e)/max(min(e),1e-9):,.0f}x   "
          f"p99.9 spread {max(l)/max(min(l),1e-9):,.0f}x")


if __name__ == "__main__":
    main()
