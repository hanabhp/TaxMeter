#!/usr/bin/env python3
"""A cost-aware cascade: the greedy scan decides when the choice is easy, LSTM-Attn when it is not.

'Easy' means the two strongest candidates differ by at least tau dB in current RSRP. Accuracy
is measured on the twenty-city live workload (results/live20 run); energy per decision is the
greedy scan's cost plus the escalation rate times LSTM-Attn's cost, both from the live run on
the same host (15 W proxy). Output: results/cascade_live20.csv
"""
import os, sys, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from taxmeter.predictors import build_events, load_predictors

d = dict(np.load(os.path.join(ROOT, "data", "workload_live20.npz")))
events, _ = build_events(d)
preds = {p.name: p for p in load_predictors(os.path.join(ROOT, "weights_live"))}
y = d["label"]
g = np.array([preds["Greedy"].predict(e) for e in events])
l = np.array([preds["LSTM-Attn"].predict(e) for e in events])
now = np.where(d["c_mask"], d["c_rh"][:, :, -1], -1e9)
top2 = -np.sort(-now, axis=1)[:, :2]
margin = top2[:, 0] - top2[:, 1]                       # dB between the two strongest candidates

cost = {r["predictor"]: float(r["E_uj_sat"]) for r in csv.DictReader(open(os.path.join(ROOT, "results", "table1_devgrade_live.csv")))}
eg, el = cost["Greedy"], cost["LSTM-Attn"]
acc_g, acc_l = float(np.mean(g == y)), float(np.mean(l == y))
rows = []
for tau in [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 1e9]:
    esc = margin < tau
    pred = np.where(esc, l, g)
    acc = float(np.mean(pred == y)); f = float(np.mean(esc))
    e = eg + f * el
    gain = (acc - acc_g) / (acc_l - acc_g) if acc_l > acc_g else float("nan")
    rows.append(dict(tau_db=tau if tau < 1e8 else "always", escalated=round(f, 4), accuracy=round(acc, 4),
                     share_of_gain=round(gain, 3), energy_uj=round(e, 1), share_of_lstm_energy=round(e / el, 3)))
    print(f"tau {str(tau if tau<1e8 else 'always'):>6}: escalate {f:6.1%}  acc {acc:.3f}  gain {gain:6.1%}  energy {e:7.1f} uJ = {e/el:6.1%} of LSTM")
print(f"greedy alone {acc_g:.3f} at {eg:.1f} uJ | LSTM alone {acc_l:.3f} at {el:.1f} uJ | events {len(y)}")
with open(os.path.join(ROOT, "results", "cascade_live20.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# ---- second stage choice: let the geometric rule (PreHO-Geo) decide when its own predicted margin is clear
from taxmeter.core import rsrp_deterministic, slant_range_km
el5 = d["c_el"] + 5.0 * d["c_rt"]
score = rsrp_deterministic(np.maximum(el5, 3.0), slant_range_km(el5))
score = np.where(d["c_mask"] & (el5 > 3.0), score, -1e9)
gp = np.argmax(score, axis=1)
check = np.array([preds["PreHO-Geo"].predict(e) for e in events[:3000]])
print(f"\nvectorized geometric rule agrees with the predictor on {np.mean(check == gp[:3000]):.3f} of 3,000 events")
t2 = -np.sort(-score, axis=1)[:, :2]; gmargin = t2[:, 0] - t2[:, 1]
eg2 = cost["PreHO-Geo"]; acc_p = float(np.mean(gp == y))
rows2 = []
for tau in [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 1e9]:
    esc = gmargin < tau; pred = np.where(esc, l, gp)
    acc = float(np.mean(pred == y)); f = float(np.mean(esc)); e = eg2 + f * el
    gain = (acc - acc_g) / (acc_l - acc_g)
    rows2.append(dict(tau_db=tau if tau < 1e8 else "always", escalated=round(f, 4), accuracy=round(acc, 4),
                      share_of_gain=round(gain, 3), energy_uj=round(e, 1), share_of_lstm_energy=round(e / el, 3)))
    print(f"geo tau {str(tau if tau<1e8 else 'always'):>6}: escalate {f:6.1%}  acc {acc:.3f}  gain over comparator {gain:6.1%}  energy {e:7.1f} uJ = {e/el:6.1%} of LSTM")
print(f"geometric rule alone {acc_p:.3f} ({(acc_p-acc_g)/(acc_l-acc_g):.0%} of the gain) at {eg2:.1f} uJ = {eg2/el:.1%} of LSTM")
with open(os.path.join(ROOT, "results", "cascade_geo_live20.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows2[0].keys()); w.writeheader(); w.writerows(rows2)
