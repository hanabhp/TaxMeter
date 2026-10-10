"""taxmeter.cli — the paper's command surface.

    python -m taxmeter.cli measure --predictor lstm-attn \
        --scenario satellite --output report.csv
"""
import argparse
import os
import sys

import numpy as np

from . import measure
from .predictors import build_events, load_predictors

ALIASES = {"3gpp-a3": "3GPP-A3", "a3": "3GPP-A3", "greedy": "Greedy",
           "kalman": "Kalman", "preho-geo": "PreHO-Geo", "preho": "PreHO-Geo",
           "phandover-mlp": "PHandover-MLP", "mlp": "PHandover-MLP",
           "dho-drl": "DHO-DRL", "drl": "DHO-DRL",
           "lstm-attn": "LSTM-Attn", "lstm": "LSTM-Attn",
           "gnn-drl": "GNN-DRL", "gnn": "GNN-DRL", "all": "all"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="taxmeter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--predictor", required=True,
                   help="name or alias, or 'all'")
    m.add_argument("--scenario", default="satellite",
                   choices=list(measure.SCENARIOS))
    m.add_argument("--workload", default="data/workload.npz")
    m.add_argument("--weights", default="weights")
    m.add_argument("--output", default="report.csv")
    a = ap.parse_args(argv)

    if not os.path.exists(a.workload):
        sys.exit(f"workload not found: {a.workload} (run run_all.py first)")
    want = ALIASES.get(a.predictor.lower())
    if want is None:
        sys.exit(f"unknown predictor '{a.predictor}'")

    d = dict(np.load(a.workload))
    events, _ = build_events(d)
    rows = []
    for p in load_predictors(a.weights):
        if want != "all" and p.name != want:
            continue
        cost = measure.measure_cost(p, events)
        ax = measure.pt_axes(cost, a.scenario)
        rows.append(dict(predictor=p.name, scenario=a.scenario,
                         mean_us=round(cost["mean_s"] * 1e6, 1),
                         p999_ms=round(cost["p999_ms"], 3),
                         mem_mb=round(cost["mem_mb"], 3),
                         E_uj=round(ax["E_uj"], 1),
                         D_orbit_usd=f'{ax["D_orbit"]:.2e}',
                         C_g_gb=f'{ax["C_g_gb"]:.2e}',
                         PT_star=round(ax["PT_star"], 2),
                         binding=ax["binding"]))
        print(f"{p.name:14s} PT*({a.scenario}) = {rows[-1]['PT_star']}%  "
              f"binding {rows[-1]['binding']}")
    if not rows:
        sys.exit("no predictor matched")
    measure.write_csv(a.output, rows)


if __name__ == "__main__":
    main()
