#!/usr/bin/env python3
"""Cost of the per-candidate stages as the candidate set grows from the five the papers
assume to the 14 to 75 satellites visible above 25 degrees on the live constellation.
MLP: per-candidate scorer, linear in K. LSTM-Attn: candidate embedding, K/V projections
and scores, linear in K (the LSTM over the RSRP sequence is fixed and excluded).
GNN-DRL: pairwise attention over K+1 nodes, quadratic in K. Same trained weights."""
import os, sys, time, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from taxmeter.predictors import load_predictors, _relu

preds = {p.name: p for p in load_predictors(os.path.join(ROOT, "weights"))}
mlp, lstm, gnn = preds["PHandover-MLP"], preds["LSTM-Attn"], preds["GNN-DRL"]
rng = np.random.default_rng(0)

def time_fn(fn, warm=50, reps=400):
    for _ in range(warm): fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns(); fn(); t[i] = time.perf_counter_ns() - t0
    return float(np.median(t) / 1e3)      # median microseconds

rows = []
for K in (5, 14, 43, 75):
    f_mlp = rng.standard_normal((K, 26)); mask = np.ones(K, bool)
    stat = rng.standard_normal((K, 4)); h = rng.standard_normal(64)
    x_g = rng.standard_normal((K + 1, 12))
    def run_mlp(): mlp.logits(f_mlp, mask)
    def run_attn():
        e = _relu(stat @ lstm.We + lstm.be); q = lstm.Wq.T @ h + lstm.bq
        k = e @ lstm.Wk + lstm.bk; v = e @ lstm.Wv + lstm.bv
        att = np.einsum("hd,jhd->j", q.reshape(2, 32), k.reshape(K, 2, 32)) / np.sqrt(32)
        return att + (v @ lstm.wo).squeeze(-1)
    def run_gnn():
        hh = x_g
        for layer in gnn.L: hh = gnn._layer(hh, *layer)
        return hh[1:] @ gnn.wo
    rows.append(dict(K=K, mlp_us=time_fn(run_mlp), attn_stage_us=time_fn(run_attn), gnn_us=time_fn(run_gnn)))
    print(rows[-1])
base = rows[0]
for r in rows:
    r["mlp_x"] = round(r["mlp_us"] / base["mlp_us"], 1); r["attn_x"] = round(r["attn_stage_us"] / base["attn_stage_us"], 1); r["gnn_x"] = round(r["gnn_us"] / base["gnn_us"], 1)
with open(os.path.join(ROOT, "results", "candidate_scaling_devgrade.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("multipliers vs K=5:", [(r["K"], r["mlp_x"], r["attn_x"], r["gnn_x"]) for r in rows])
