#!/usr/bin/env python3
"""Does the cost spread survive batching?

A satellite or gateway serving many terminals can score many decisions in one call. This
script times every predictor at batch sizes 1, 8, 32, 128 and 512, single-threaded as in the
paper: the four learned predictors through their PyTorch modules (same weights as the NumPy
path), the rules as vectorized NumPy. Batched decisions are checked against the one-at-a-time
NumPy predictors before any timing is reported. Output: results/batching_devgrade.csv
"""
import os, sys, time, csv
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from taxmeter.predictors import build_events, load_predictors
from taxmeter.core import rsrp_deterministic, slant_range_km
import runtime_check as rc                       # reuses its weight loading for the torch modules

torch.set_num_threads(1)
events, _ = rc.events, None
np_preds = rc.np_preds
rng = np.random.default_rng(11)
POOL = rng.choice(len(events), 2048, replace=False)

def stack(idx, attr, dtype=np.float32):
    return np.stack([getattr(events[i], attr) for i in idx]).astype(dtype)

# ---- batched rules (vectorized NumPy, same arithmetic as the per-event rules)
def greedy_b(B):
    now = np.where(B["mask"], B["crh"][:, :, -1], -1e9); return np.argmax(now, axis=1)
def a3_b(B):
    now = np.where(B["mask"], B["crh"][:, :, -1], -1e9); i = np.argmax(now, axis=1)
    _trigger = now[np.arange(len(i)), i] > B["srh"][:, -1] + 3.0
    return i
def preho_b(B):
    el5 = B["cel"] + 5.0 * B["crt"]
    score = rsrp_deterministic(np.maximum(el5, 3.0), slant_range_km(el5))
    return np.argmax(np.where(B["mask"] & (el5 > 3.0), score, -1e9), axis=1)
KV = np_preds["Kalman-vec"]
def kalman_b(B):
    z = B["ceh"].reshape(-1, B["ceh"].shape[-1]); n, T = z.shape
    x = np.stack([z[:, 0], np.zeros(n)], axis=1); P = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    F, Q, R = KV.F, KV.Q, KV.R
    for t in range(1, T):
        x = x @ F.T; P = F @ P @ F.T + Q; y = z[:, t] - x[:, 0]
        s_ = P[:, 0, 0] + R; k = P[:, :, 0] / s_[:, None]
        x = x + k * y[:, None]; P = P - k[:, :, None] * P[:, None, 0, :]
    el5 = (x[:, 0] + 5.0 * x[:, 1]).reshape(B["ceh"].shape[0], -1)
    score = rsrp_deterministic(np.maximum(el5, 3.0), slant_range_km(el5))
    return np.argmax(np.where(B["mask"] & (el5 > 3.0), score, -1e9), axis=1)

def torch_fn(name):
    with torch.no_grad():
        if name == "PHandover-MLP": return lambda B: rc.mlp(B["t_mlp"]).argmax(-1)
        if name == "DHO-DRL":       return lambda B: rc.drl(B["t_drl"]).argmax(-1)
        if name == "LSTM-Attn":     return lambda B: rc.lstm(B["t_seq"], B["t_stat"]).argmax(-1)
        if name == "GNN-DRL":       return lambda B: rc.gnn(B["t_gnn"]).argmax(-1)

FNS = {"3GPP-A3": a3_b, "Greedy": greedy_b, "Kalman-vec": kalman_b, "PreHO-Geo": preho_b,
       "PHandover-MLP": torch_fn("PHandover-MLP"), "DHO-DRL": torch_fn("DHO-DRL"),
       "LSTM-Attn": torch_fn("LSTM-Attn"), "GNN-DRL": torch_fn("GNN-DRL")}

def make_batch(idx):
    B = dict(mask=stack(idx, "c_mask", bool), crh=stack(idx, "c_rh", np.float64), srh=stack(idx, "s_rh", np.float64),
             cel=stack(idx, "c_el", np.float64), crt=stack(idx, "c_rt", np.float64), ceh=stack(idx, "c_eh", np.float64))
    B.update(t_mlp=torch.from_numpy(stack(idx, "f_mlp")), t_drl=torch.from_numpy(stack(idx, "f_drl")),
             t_seq=torch.from_numpy(stack(idx, "f_seq")), t_stat=torch.from_numpy(stack(idx, "f_stat")),
             t_gnn=torch.from_numpy(stack(idx, "f_gnn")))
    return B

# ---- correctness: batched decisions vs the per-event NumPy predictors
check = make_batch(POOL[:256])
ref_name = {"Kalman-vec": "Kalman-vec"}
agree = {}
for name, fn in FNS.items():
    got = np.asarray(fn(check)).reshape(-1)
    ref = np.array([np_preds[ref_name.get(name, name)].predict(events[i]) for i in POOL[:256]])
    agree[name] = float(np.mean(got == ref))
print("decision agreement with the per-event predictors:", {k: round(v, 3) for k, v in agree.items()})

# ---- timing
rows = []
for bs in (1, 8, 32, 128, 512):
    batches = [make_batch(POOL[(j * bs) % (2048 - bs):(j * bs) % (2048 - bs) + bs]) for j in range(8)]
    for name, fn in FNS.items():
        for b in batches[:2]: fn(b)                     # warm-up
        reps, samples = 0, []
        t_end = time.perf_counter() + 0.6
        while time.perf_counter() < t_end or reps < 20:
            b = batches[reps % len(batches)]
            t0 = time.perf_counter_ns(); fn(b); samples.append(time.perf_counter_ns() - t0); reps += 1
        per = np.median(samples) / 1e3 / bs
        rows.append(dict(batch=bs, predictor=name, per_decision_us=round(per, 4),
                         call_ms=round(np.median(samples) / 1e6, 4), agreement=round(agree[name], 3)))
    print(f"batch {bs:4d}: " + "  ".join(f"{r['predictor'][:6]} {r['per_decision_us']:.3f}" for r in rows if r["batch"] == bs))
with open(os.path.join(ROOT, "results", "batching_devgrade.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for bs in (1, 32, 512):
    v = {r["predictor"]: r["per_decision_us"] for r in rows if r["batch"] == bs}
    print(f"batch {bs}: spread {max(v.values()) / min(v.values()):.0f}x | LSTM {v['LSTM-Attn']:.2f} us = "
          f"{v['LSTM-Attn'] * 15:.1f} uJ at 15 W | GNN {v['GNN-DRL']:.2f} us | Greedy {v['Greedy']:.4f} us")
