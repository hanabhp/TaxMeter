"""taxmeter.predictors — eight predictors, pure-NumPy inference.

Stated parameter counts (paper): MLP 9,473 | DHO-DRL 27,137 |
LSTM-Attn 31,041 | GNN-DRL 4,929.  Rebuilt counts are printed by
param_report(); LSTM-Attn lands EXACTLY on 31,041 with LSTM(6->64,
combined bias) + Wq/Wk/Wv (64x64) + cand-embed(4->64) + combine(64->1),
which strongly constrains what the original architecture must have been.
"""
import numpy as np
from .core import slant_range_km, rsrp_deterministic

# ----------------------------- event container ------------------------------

class Event:
    __slots__ = ("s_rh", "s_eh", "s_el", "s_rt", "s_rg",
                 "c_rh", "c_eh", "c_el", "c_rt", "c_rg", "c_mask",
                 "f_mlp", "f_drl", "f_seq", "f_stat", "f_gnn", "label")

def _norm_rh(rh):   return (rh + 90.0) / 20.0
def _norm_el(el):   return el / 90.0
def _norm_rt(rt):   return rt * 2.0
def _norm_rg(rg):   return rg / 2000.0

def featurize_all(d):
    """Batched featurizers over the whole workload dict (single source of
    truth for both torch training and NumPy inference)."""
    N = len(d["label"])
    s_rh = _norm_rh(d["s_rh"]); s_eh = _norm_el(d["s_eh"])
    c_rh = _norm_rh(d["c_rh"]); c_eh = _norm_el(d["c_eh"])
    s_el = _norm_el(d["s_el"]); s_rt = _norm_rt(d["s_rt"]); s_rg = _norm_rg(d["s_rg"])
    c_el = _norm_el(d["c_el"]); c_rt = _norm_rt(d["c_rt"]); c_rg = _norm_rg(d["c_rg"])
    mask = d["c_mask"].astype(bool)
    s_now = s_rh[:, -1]; c_now = c_rh[:, :, -1]

    # MLP: per-candidate 26-dim
    f_mlp = np.concatenate([
        c_rh,                                             # 8
        np.repeat(s_rh[:, None, :], 5, axis=1),           # 8
        c_el[..., None], c_rt[..., None], c_rg[..., None],            # 3
        np.repeat(s_el[:, None, None], 5, 1),
        np.repeat(s_rt[:, None, None], 5, 1),
        np.repeat(s_rg[:, None, None], 5, 1),                          # 3
        (c_now - s_now[:, None])[..., None],
        (c_el - s_el[:, None])[..., None],
        (c_rg - s_rg[:, None])[..., None],                             # 3
        c_rh.mean(axis=2, keepdims=True),                              # 1
    ], axis=2).astype(np.float32)                          # (N,5,26)

    # DHO-DRL: event-level 119-dim
    cb = np.concatenate([c_rh, c_eh, c_el[..., None], c_rt[..., None],
                         c_rg[..., None], (c_now - s_now[:, None])[..., None]],
                        axis=2)                            # (N,5,20)
    cb = cb * mask[..., None]
    f_drl = np.concatenate([s_rh, s_eh, s_el[:, None], s_rt[:, None],
                            s_rg[:, None], cb.reshape(N, -1)],
                           axis=1).astype(np.float32)      # (N,119)

    # LSTM-Attn: sequence (N,8,6) + candidate statics (N,5,4)
    seq = np.concatenate([s_rh[:, :, None],
                          np.transpose(c_rh, (0, 2, 1))], axis=2)
    m = np.repeat(mask[:, None, :], seq.shape[1], axis=1)
    seq[:, :, 1:][~m] = -1.0
    f_seq = seq.astype(np.float32)
    f_stat = np.stack([c_el, c_rt, c_rg, c_now], axis=2).astype(np.float32)

    # GNN: 6 nodes x 12 feats (node 0 = serving)
    def node_feats(now, elh, rh, el, rt, rg, is_srv, dn, de, dr, mk):
        slope_r = (rh[..., -1] - rh[..., 0]) / 7.0
        slope_e = (elh[..., -1] - elh[..., 0]) / 7.0
        return np.stack([now, el, rt, rg, rh.mean(-1), slope_r, slope_e,
                         is_srv, dn, de, dr, mk], axis=-1)
    z = np.zeros(N, np.float32); o = np.ones(N, np.float32)
    srv = node_feats(s_now, s_eh, s_rh, s_el, s_rt, s_rg, o, z, z, z, o)
    cnd = node_feats(c_now, c_eh, c_rh, c_el, c_rt, c_rg,
                     np.zeros((N, 5), np.float32),
                     c_now - s_now[:, None], c_el - s_el[:, None],
                     c_rg - s_rg[:, None], mask.astype(np.float32))
    f_gnn = np.concatenate([srv[:, None, :], cnd], axis=1).astype(np.float32)

    return dict(f_mlp=f_mlp, f_drl=f_drl, f_seq=f_seq, f_stat=f_stat,
                f_gnn=f_gnn, mask=mask)

def build_events(d):
    feats = featurize_all(d)
    N = len(d["label"])
    evs = []
    for i in range(N):
        e = Event()
        for k in ("s_rh", "s_eh", "c_rh", "c_eh", "c_el", "c_rt", "c_rg"):
            setattr(e, k, np.ascontiguousarray(d[k][i], dtype=np.float64))
        e.s_el = float(d["s_el"][i]); e.s_rt = float(d["s_rt"][i])
        e.s_rg = float(d["s_rg"][i])
        e.c_mask = d["c_mask"][i].astype(bool)
        e.f_mlp = np.ascontiguousarray(feats["f_mlp"][i], np.float64)
        e.f_drl = np.ascontiguousarray(feats["f_drl"][i], np.float64)
        e.f_seq = np.ascontiguousarray(feats["f_seq"][i], np.float64)
        e.f_stat = np.ascontiguousarray(feats["f_stat"][i], np.float64)
        e.f_gnn = np.ascontiguousarray(feats["f_gnn"][i], np.float64)
        e.label = int(d["label"][i])
        evs.append(e)
    return evs, feats

# ------------------------------- baselines ----------------------------------

class GPP_A3:
    name, params = "3GPP-A3", 0
    def __init__(self, hys_db=3.0): self.hys = hys_db
    def predict(self, ev):
        now = ev.c_rh[:, -1]
        srv = ev.s_rh[-1] + self.hys
        cand = np.where(ev.c_mask, now, -1e9)
        i = int(np.argmax(cand))
        _trigger = cand[i] > srv          # fixed-form comparison w/ hysteresis
        return i

class Greedy:
    name, params = "Greedy", 0
    def predict(self, ev):
        best, bi = -1e9, 0
        now = ev.c_rh[:, -1]
        for j in range(5):                 # literal max-RSRP scan
            if ev.c_mask[j] and now[j] > best:
                best, bi = now[j], j
        return bi

class Kalman:
    """Constant-velocity KF on candidate elevation history; predicts the
    deterministic RSRP at t+5 s from geometry (ephemeris advantage)."""
    name, params = "Kalman", 0
    def __init__(self, dt=10.0, q=1e-4, r=0.05 ** 2):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2],
                               [dt ** 2 / 2, dt]])
        self.R = r
    def predict(self, ev):
        best, bi = -1e9, 0
        for j in range(5):
            if not ev.c_mask[j]:
                continue
            z = ev.c_eh[j]
            x = np.array([z[0], 0.0]); P = np.eye(2)
            for t in range(1, len(z)):
                x = self.F @ x
                P = self.F @ P @ self.F.T + self.Q
                y = z[t] - x[0]
                s = P[0, 0] + self.R
                k = P[:, 0] / s
                x = x + k * y
                P = P - np.outer(k, P[0, :])
            el5 = x[0] + 5.0 * x[1]
            if el5 < 3.0:
                score = -1e9
            else:
                score = rsrp_deterministic(el5, slant_range_km(el5))
            if score > best:
                best, bi = score, j
        return bi


class KalmanVec:
    """Same constant-velocity KF as Kalman, vectorized across the five
    candidates (one small-matrix recursion instead of a Python loop per
    candidate). Added to test how much of Kalman's cost is the literal loop."""
    name, params = "Kalman-vec", 0
    def __init__(self, dt=10.0, q=1e-4, r=0.05 ** 2):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.R = r
    def predict(self, ev):
        z = ev.c_eh                                   # (5, T)
        n, T = z.shape
        x = np.stack([z[:, 0], np.zeros(n)], axis=1)  # (5, 2)
        P = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
        F, Q, R = self.F, self.Q, self.R
        for t in range(1, T):
            x = x @ F.T
            P = F @ P @ F.T + Q
            y = z[:, t] - x[:, 0]
            s_ = P[:, 0, 0] + R
            k = P[:, :, 0] / s_[:, None]
            x = x + k * y[:, None]
            P = P - k[:, :, None] * P[:, None, 0, :]
        el5 = x[:, 0] + 5.0 * x[:, 1]
        score = rsrp_deterministic(np.maximum(el5, 3.0), slant_range_km(el5))
        score = np.where(ev.c_mask & (el5 > 3.0), score, -1e9)
        return int(np.argmax(score))

class PreHOGeo:
    """Ephemeris-geometric: linear elevation extrapolation to t+5 s."""
    name, params = "PreHO-Geo", 0
    def predict(self, ev):
        el5 = ev.c_el + 5.0 * ev.c_rt
        score = rsrp_deterministic(np.maximum(el5, 3.0),
                                   slant_range_km(el5))
        score = np.where(ev.c_mask & (el5 > 3.0), score, -1e9)
        return int(np.argmax(score))

# --------------------------- learned predictors -----------------------------

def _relu(x): return np.maximum(x, 0.0)
def _sig(x):  return 1.0 / (1.0 + np.exp(-x))

class MLPPredictor:
    """PHandover-MLP: per-candidate scorer, 26 -> 96 -> 64 -> 1."""
    name = "PHandover-MLP"
    def __init__(self, w):
        self.W1, self.b1 = w["W1"], w["b1"]
        self.W2, self.b2 = w["W2"], w["b2"]
        self.W3, self.b3 = w["W3"], w["b3"]
        self.params = sum(int(np.prod(w[k].shape)) for k in w)
    def logits(self, f, mask):
        h = _relu(f @ self.W1 + self.b1)
        h = _relu(h @ self.W2 + self.b2)
        s = (h @ self.W3 + self.b3).squeeze(-1)
        return np.where(mask, s, -1e9)
    def predict(self, ev):
        return int(np.argmax(self.logits(ev.f_mlp, ev.c_mask)))

class DRLPredictor:
    """DHO-DRL: event-level policy net, 119 -> 128 -> 96 -> 6."""
    name = "DHO-DRL"
    def __init__(self, w):
        self.W1, self.b1 = w["W1"], w["b1"]
        self.W2, self.b2 = w["W2"], w["b2"]
        self.W3, self.b3 = w["W3"], w["b3"]
        self.params = sum(int(np.prod(w[k].shape)) for k in w)
    def logits(self, f, mask):
        h = _relu(f @ self.W1 + self.b1)
        h = _relu(h @ self.W2 + self.b2)
        q = h @ self.W3 + self.b3
        return np.where(mask, q[..., :5], -1e9)
    def predict(self, ev):
        return int(np.argmax(self.logits(ev.f_drl, ev.c_mask)))

class LSTMAttnPredictor:
    """LSTM(6->64) over the RSRP sequence + two attention heads over the
    candidate set.  31,041 parameters exactly (paper's stated count)."""
    name = "LSTM-Attn"
    N_HEADS, DK = 2, 32
    def __init__(self, w):
        self.Wih, self.Whh, self.b = w["Wih"], w["Whh"], w["b"]   # (256,6),(256,64),(256,)
        self.We, self.be = w["We"], w["be"]                        # (4,64),(64,)
        self.Wq, self.bq = w["Wq"], w["bq"]
        self.Wk, self.bk = w["Wk"], w["bk"]
        self.Wv, self.bv = w["Wv"], w["bv"]
        self.wo, self.bo = w["wo"], w["bo"]                        # (64,1),(1,)
        self.params = sum(int(np.prod(w[k].shape)) for k in w)
    def _lstm(self, seq):                       # seq (T,6) -> h (64,)
        h = np.zeros(64); c = np.zeros(64)
        for t in range(seq.shape[0]):
            z = self.Wih @ seq[t] + self.Whh @ h + self.b
            i = _sig(z[0:64]); f = _sig(z[64:128])
            g = np.tanh(z[128:192]); o = _sig(z[192:256])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h
    def logits(self, seq, stat, mask):
        h = self._lstm(seq)
        e = _relu(stat @ self.We + self.be)                 # (5,64)
        q = self.Wq.T @ h + self.bq                         # (64,)
        k = e @ self.Wk + self.bk                           # (5,64)
        v = e @ self.Wv + self.bv                           # (5,64)
        qh = q.reshape(self.N_HEADS, self.DK)
        kh = k.reshape(5, self.N_HEADS, self.DK)
        att = np.einsum("hd,jhd->j", qh, kh) / np.sqrt(self.DK)
        s = att + (v @ self.wo).squeeze(-1) + self.bo[0]
        return np.where(mask, s, -1e9)
    def predict(self, ev):
        return int(np.argmax(self.logits(ev.f_seq, ev.f_stat, ev.c_mask)))

class GNNPredictor:
    """GNN-DRL: two GATv2 layers with a root term over the 6-node complete graph
    (serving + 5 candidates), 12 -> 36 -> 20 -> 1. Pairwise GATv2 attention plus
    the node's own transformed features; without the root term every candidate
    aggregates the same mixture and gets the same score, a degeneracy the runtime
    check caught in the first build."""
    name = "GNN-DRL"
    def __init__(self, w):
        self.L = [(w["W1"], w["b1"], w["a1"], w["V1"], w["c1"], w["U1"], w["d1"]),
                  (w["W2"], w["b2"], w["a2"], w["V2"], w["c2"], w["U2"], w["d2"])]
        self.wo, self.bo = w["wo"], w["bo"]
        self.params = sum(int(np.prod(w[k].shape)) for k in w)
    @staticmethod
    def _layer(x, W, b, a, V, c, U, d):
        n = x.shape[0]
        pair = np.concatenate([np.repeat(x, n, 0), np.tile(x, (n, 1))], axis=1)   # (n*n, 2F)
        z = pair @ W + b
        z = np.where(z > 0, z, 0.2 * z)
        e = (z @ a).reshape(n, n)
        e = e - e.max(axis=1, keepdims=True)
        al = np.exp(e); al /= al.sum(axis=1, keepdims=True)
        out = (x @ U + d) + al @ (x @ V + c)                                       # root + message
        return np.where(out > 0, out, np.exp(np.minimum(out, 0.0)) - 1.0)         # ELU
    def logits(self, x, mask):
        h = x
        for layer in self.L:
            h = self._layer(h, *layer)
        s = (h[1:6] @ self.wo).squeeze(-1) + self.bo[0]
        return np.where(mask, s, -1e9)
    def predict(self, ev):
        return int(np.argmax(self.logits(ev.f_gnn, ev.c_mask)))

# --------------------------------- registry ---------------------------------

STATED_PARAMS = {"3GPP-A3": 0, "Greedy": 0, "Kalman": 0, "Kalman-vec": 0, "PreHO-Geo": 0,
                 "PHandover-MLP": 9473, "DHO-DRL": 27137,
                 "LSTM-Attn": 31041, "GNN-DRL": 4929}

def load_predictors(weights_dir="weights"):
    import os
    def w(name):
        return dict(np.load(os.path.join(weights_dir, name + ".npz")))
    return [GPP_A3(), Greedy(), Kalman(), KalmanVec(), PreHOGeo(),
            MLPPredictor(w("mlp")), DRLPredictor(w("drl")),
            LSTMAttnPredictor(w("lstm")), GNNPredictor(w("gnn"))]

def param_report(preds):
    rows = []
    for p in preds:
        stated = STATED_PARAMS.get(p.name, 0)
        rows.append((p.name, stated, p.params))
    return rows
