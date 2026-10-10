#!/usr/bin/env python3
"""Two host-independent complements to the TDP proxy.

(1) Analytic multiply-accumulate counts per decision for every predictor, so the
    ordering claim can be checked against arithmetic rather than wall-clock time.
(2) Forward-pass time of the per-candidate predictors as the candidate set grows
    from 5 (the papers' cap) to 75 (the most satellites visible above 25 degrees in
    the live run), measured with the paper's protocol on the same host.
"""
import csv, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from taxmeter.predictors import load_predictors, Event

# ---- (1) MAC counts (dense layers: in*out; LSTM: 4*h*(in+h) per step; GAT pair scores)
def macs(preds):
    """MAC counts per decision derived from the loaded weight shapes."""
    out = {"3GPP-A3": 10, "Greedy": 5, "PreHO-Geo": 60, "Kalman": 5*7*30, "Kalman-vec": 5*7*30}
    m = preds["PHandover-MLP"]; out["PHandover-MLP"] = 5 * (m.W1.size + m.W2.size + m.W3.size)
    d = preds["DHO-DRL"]; out["DHO-DRL"] = d.W1.size + d.W2.size + d.W3.size
    l = preds["LSTM-Attn"]; out["LSTM-Attn"] = 8 * (l.Wih.size + l.Whh.size) + 5*l.We.size + l.Wq.size + 2*5*l.Wk.size + 5*l.wo.size
    g = preds["GNN-DRL"]; n = 6; tot = 0
    for (W, b, a, V, c, U, dd) in g.L:
        tot += n*n*W.size + n*n*a.size + n*V.size + n*U.size
    out["GNN-DRL"] = tot + 5*g.wo.size
    return out

# ---- (2) candidate scaling for the predictors whose cost depends on K
def scaled_event(base, K, rng):
    e = Event()
    for k in ("s_rh","s_eh"): setattr(e, k, base.__getattribute__(k))
    e.s_el, e.s_rt, e.s_rg = base.s_el, base.s_rt, base.s_rg
    rep = int(np.ceil(K/5))
    def tile(a): return np.ascontiguousarray(np.concatenate([a]*rep, axis=0)[:K] + 0.0)
    e.c_rh, e.c_eh = tile(base.c_rh), tile(base.c_eh)
    e.c_el, e.c_rt, e.c_rg = tile(base.c_el), tile(base.c_rt), tile(base.c_rg)
    e.c_mask = np.ones(K, bool)
    e.f_mlp = tile(base.f_mlp)                       # (K,26)
    e.f_stat = tile(base.f_stat)                     # (K,4)
    e.f_seq = base.f_seq                             # sequence input is fixed (serving + 5)
    srv = base.f_gnn[:1]; cand = tile(base.f_gnn[1:]) # (K,12)
    e.f_gnn = np.ascontiguousarray(np.concatenate([srv, cand], axis=0))
    e.f_drl = base.f_drl
    return e

class _GNNK:  # GNN scorer over K candidate nodes (same weights)
    def __init__(self, g): self.g = g
    def predict(self, ev):
        g = self.g; h = ev.f_gnn
        for layer in g.L: h = g._layer(h, *layer)
        s = (h[1:] @ g.wo).squeeze(-1) + g.bo[0]
        return int(np.argmax(np.where(ev.c_mask, s, -1e9)))

class _LSTMK:  # attention over K keys (same weights)
    def __init__(self, l): self.l = l
    def predict(self, ev):
        l = self.l; h = l._lstm(ev.f_seq)
        e = np.maximum(ev.f_stat @ l.We + l.be, 0.0)
        q = l.Wq.T @ h + l.bq; k = e @ l.Wk + l.bk; v = e @ l.Wv + l.bv
        att = (k.reshape(len(k), 2, 32) * q.reshape(2, 32)).sum(axis=(1, 2)) / np.sqrt(32.0)
        s = att + (v @ l.wo).squeeze(-1) + l.bo[0]
        return int(np.argmax(np.where(ev.c_mask, s, -1e9)))

def timeit(pred, ev, warm=50, n=400):
    for _ in range(warm): pred.predict(ev)
    t = np.empty(n); pc = time.perf_counter_ns
    for i in range(n):
        t0 = pc(); pred.predict(ev); t[i] = pc() - t0
    return float(np.median(t) / 1e3)   # us

def main():
    preds = {p.name: p for p in load_predictors(os.path.join(ROOT, "weights"))}
    d = dict(np.load(os.path.join(ROOT, "data", "workload.npz")))
    from taxmeter.predictors import build_events
    events, _ = build_events(d)
    rng = np.random.default_rng(3)
    base = [events[i] for i in rng.choice(len(events), 20, replace=False) if events[i].c_mask.all()][:10]
    m = macs(preds)
    rows = []
    for name, p in preds.items():
        us = np.median([timeit(p, e, 20, 100) for e in base])
        rows.append(dict(predictor=name, macs=m[name], median_us=round(us, 2), ns_per_mac=round(1e3*us/m[name], 2)))
        print(f"{name:14s} MACs {m[name]:8d}  median {us:8.1f} us  ns/MAC {1e3*us/m[name]:9.2f}")
    with open(os.path.join(ROOT, "results", "op_counts_devgrade.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    scal = {"PHandover-MLP": preds["PHandover-MLP"], "LSTM-Attn": _LSTMK(preds["LSTM-Attn"]), "GNN-DRL": _GNNK(preds["GNN-DRL"])}
    Ks = [5, 10, 20, 40, 75]
    out = []
    print("\ncandidate scaling (median us), K =", Ks)
    for name, p in scal.items():
        t = [np.median([timeit(p, scaled_event(e, K, rng), 20, 100) for e in base]) for K in Ks]
        out.append(dict(predictor=name, **{f"K{K}": round(v, 2) for K, v in zip(Ks, t)}, x75_over_x5=round(t[-1]/t[0], 2)))
        print(f"{name:14s} " + " ".join(f"{v:8.1f}" for v in t) + f"   x{t[-1]/t[0]:.1f}")
    with open(os.path.join(ROOT, "results", "candidate_scaling_devgrade.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    main()
