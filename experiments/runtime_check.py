#!/usr/bin/env python3
"""Does the cost ordering of the four learned predictors survive a different runtime?
Times a PyTorch (CPU, single thread) forward pass for each learned predictor under the
paper's protocol (300 events, 20 warm-up, 50 timed calls) and compares with the NumPy path."""
import os, sys, time, csv
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from taxmeter import measure
from taxmeter.predictors import build_events, load_predictors
from taxmeter.train_torch import MLPNet, DRLNet, LSTMAttnNet, GNNNet

torch.set_num_threads(1)
d = dict(np.load(os.path.join(ROOT, "data", "workload.npz")))
events, _ = build_events(d)
np_preds = {p.name: p for p in load_predictors(os.path.join(ROOT, "weights"))}

def load_torch(name, cls, keymap):
    w = dict(np.load(os.path.join(ROOT, "weights", name + ".npz")))
    m = cls()
    sd = m.state_dict()
    for tk, (nk, transpose) in keymap.items():
        arr = torch.tensor(w[nk], dtype=torch.float32)
        sd[tk] = arr.T.contiguous() if transpose else arr
    m.load_state_dict(sd); m.eval(); return m

mlp = load_torch("mlp", MLPNet, {"l1.weight": ("W1", True), "l1.bias": ("b1", False), "l2.weight": ("W2", True),
                                 "l2.bias": ("b2", False), "l3.weight": ("W3", True), "l3.bias": ("b3", False)})
drl = load_torch("drl", DRLNet, {"l1.weight": ("W1", True), "l1.bias": ("b1", False), "l2.weight": ("W2", True),
                                 "l2.bias": ("b2", False), "l3.weight": ("W3", True), "l3.bias": ("b3", False)})
gnn = load_torch("gnn", GNNNet, {"W1.weight": ("W1", True), "W1.bias": ("b1", False), "a1": ("a1", False), "V1.weight": ("V1", True), "V1.bias": ("c1", False), "U1.weight": ("U1", True), "U1.bias": ("d1", False),
                                 "W2.weight": ("W2", True), "W2.bias": ("b2", False), "a2": ("a2", False), "V2.weight": ("V2", True), "V2.bias": ("c2", False), "U2.weight": ("U2", True), "U2.bias": ("d2", False),
                                 "o.weight": ("wo", True), "o.bias": ("bo", False)})
# LSTM: torch keeps two biases; split the combined one in half (values only affect correctness, not cost)
w = dict(np.load(os.path.join(ROOT, "weights", "lstm.npz")))
lstm = LSTMAttnNet(); sd = lstm.state_dict()
sd["lstm.weight_ih_l0"] = torch.tensor(w["Wih"], dtype=torch.float32); sd["lstm.weight_hh_l0"] = torch.tensor(w["Whh"], dtype=torch.float32)
sd["lstm.bias_ih_l0"] = torch.tensor(w["b"], dtype=torch.float32); sd["lstm.bias_hh_l0"] = torch.zeros(256)
for tk, nk, tr in [("emb.weight","We",True),("emb.bias","be",False),("q.weight","Wq",True),("q.bias","bq",False),
                   ("k.weight","Wk",True),("k.bias","bk",False),("v.weight","Wv",True),("v.bias","bv",False),
                   ("o.weight","wo",True),("o.bias","bo",False)]:
    arr = torch.tensor(w[nk], dtype=torch.float32); sd[tk] = arr.T.contiguous() if tr else arr
lstm.load_state_dict(sd); lstm.eval()

class TorchPred:
    params = 0
    def __init__(self, name, fn): self.name, self.fn = name, fn
    def predict(self, ev): 
        with torch.no_grad(): return int(self.fn(ev).argmax())

tp = [TorchPred("PHandover-MLP", lambda ev: mlp(torch.from_numpy(ev.f_mlp.astype(np.float32))[None])[0]),
      TorchPred("DHO-DRL",       lambda ev: drl(torch.from_numpy(ev.f_drl.astype(np.float32))[None])[0]),
      TorchPred("LSTM-Attn",     lambda ev: lstm(torch.from_numpy(ev.f_seq.astype(np.float32))[None], torch.from_numpy(ev.f_stat.astype(np.float32))[None])[0]),
      TorchPred("GNN-DRL",       lambda ev: gnn(torch.from_numpy(ev.f_gnn.astype(np.float32))[None])[0])]
if __name__ == "__main__":
    rows = []
    print(f"{'predictor':14s} {'numpy mean us':>14s} {'torch mean us':>14s} {'numpy p99.9':>12s} {'torch p99.9':>12s}  agree")
    for t in tp:
        cn = measure.measure_cost(np_preds[t.name], events, seed=7); ct = measure.measure_cost(t, events, seed=7)
        rs = np.random.default_rng(3).choice(len(events), 500, replace=False)
        agree = np.mean([np_preds[t.name].predict(events[i]) == t.predict(events[i]) for i in rs])
        rows.append(dict(predictor=t.name, numpy_mean_us=cn["mean_s"]*1e6, torch_mean_us=ct["mean_s"]*1e6,
                         numpy_p999_ms=cn["p999_ms"], torch_p999_ms=ct["p999_ms"], decision_agreement=agree))
        print(f"{t.name:14s} {cn['mean_s']*1e6:14.1f} {ct['mean_s']*1e6:14.1f} {cn['p999_ms']:12.3f} {ct['p999_ms']:12.3f}  {agree:.3f}")
    with open(os.path.join(ROOT, "results", "runtime_check_devgrade.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    order = lambda key: [r["predictor"] for r in sorted(rows, key=lambda r: r[key])]
    print("numpy order:", order("numpy_mean_us")); print("torch order:", order("torch_mean_us"))
