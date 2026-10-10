"""taxmeter.train_torch — trains the four learned predictors (torch, CPU)
and exports weights into the pure-NumPy inference layout.

SPEC GAP: the paper never states how its re-implementations were trained
(DRL environments cannot be re-run from a prose description).  We train all
four supervised on the best-target label — a documented divergence for the
two DRL-derived architectures.  Weights do not affect the cost axes; they
only affect accuracy.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(26, 96)
        self.l2 = nn.Linear(96, 64)
        self.l3 = nn.Linear(64, 1)
    def forward(self, f):                       # f (B,5,26)
        h = Fn.relu(self.l1(f))
        h = Fn.relu(self.l2(h))
        return self.l3(h).squeeze(-1)           # (B,5)
    def export(self):
        return dict(W1=self.l1.weight.T, b1=self.l1.bias,
                    W2=self.l2.weight.T, b2=self.l2.bias,
                    W3=self.l3.weight.T, b3=self.l3.bias)


class DRLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(119, 128)
        self.l2 = nn.Linear(128, 96)
        self.l3 = nn.Linear(96, 6)
    def forward(self, f):                       # f (B,119)
        h = Fn.relu(self.l1(f))
        h = Fn.relu(self.l2(h))
        return self.l3(h)[:, :5]
    def export(self):
        return dict(W1=self.l1.weight.T, b1=self.l1.bias,
                    W2=self.l2.weight.T, b2=self.l2.bias,
                    W3=self.l3.weight.T, b3=self.l3.bias)


class LSTMAttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, batch_first=True)
        self.emb = nn.Linear(4, 64)
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)
        self.v = nn.Linear(64, 64)
        self.o = nn.Linear(64, 1)
    def forward(self, seq, stat):               # (B,8,6), (B,5,4)
        _, (h, _) = self.lstm(seq)
        h = h[0]                                # (B,64)
        e = Fn.relu(self.emb(stat))             # (B,5,64)
        q = self.q(h)                           # (B,64)
        k = self.k(e); v = self.v(e)
        att = (k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(32.0)
        return att + self.o(v).squeeze(-1)
    def export(self):
        b = self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0
        return dict(Wih=self.lstm.weight_ih_l0, Whh=self.lstm.weight_hh_l0,
                    b=b,
                    We=self.emb.weight.T, be=self.emb.bias,
                    Wq=self.q.weight.T, bq=self.q.bias,
                    Wk=self.k.weight.T, bk=self.k.bias,
                    Wv=self.v.weight.T, bv=self.v.bias,
                    wo=self.o.weight.T, bo=self.o.bias)


class GNNNet(nn.Module):
    """GATv2 with root term, 12 -> 36 -> 20 -> 1, mirrors predictors.GNNPredictor."""
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(24, 36); self.a1 = nn.Parameter(torch.empty(36, 1)); self.V1 = nn.Linear(12, 36); self.U1 = nn.Linear(12, 36)
        self.W2 = nn.Linear(72, 20); self.a2 = nn.Parameter(torch.empty(20, 1)); self.V2 = nn.Linear(36, 20); self.U2 = nn.Linear(36, 20)
        self.o = nn.Linear(20, 1)
        nn.init.xavier_uniform_(self.a1); nn.init.xavier_uniform_(self.a2)
    @staticmethod
    def _layer(x, W, a, V, U):
        B, n, F = x.shape
        xi = x.unsqueeze(2).expand(B, n, n, F); xj = x.unsqueeze(1).expand(B, n, n, F)
        z = Fn.leaky_relu(W(torch.cat([xi, xj], dim=-1)), 0.2)
        al = torch.softmax((z @ a).squeeze(-1), dim=2)
        return Fn.elu(U(x) + al @ V(x))
    def forward(self, x):
        h = self._layer(x, self.W1, self.a1, self.V1, self.U1)
        h = self._layer(h, self.W2, self.a2, self.V2, self.U2)
        return self.o(h[:, 1:6, :]).squeeze(-1)
    def export(self):
        return dict(W1=self.W1.weight.T, b1=self.W1.bias, a1=self.a1, V1=self.V1.weight.T, c1=self.V1.bias, U1=self.U1.weight.T, d1=self.U1.bias,
                    W2=self.W2.weight.T, b2=self.W2.bias, a2=self.a2, V2=self.V2.weight.T, c2=self.V2.bias, U2=self.U2.weight.T, d2=self.U2.bias,
                    wo=self.o.weight.T, bo=self.o.bias)


def _train(model, inputs, labels, mask, tr, te, epochs, name, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(tr)
    for ep in range(epochs):
        perm = tr[torch.randperm(n)]
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            logits = model(*[x[b] for x in inputs])
            logits = logits.masked_fill(~mask[b], -1e9)
            loss = Fn.cross_entropy(logits, labels[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        def acc(ix):
            lg = model(*[x[ix] for x in inputs])
            lg = lg.masked_fill(~mask[ix], -1e9)
            return float((lg.argmax(1) == labels[ix]).float().mean())
        a_tr, a_te = acc(tr), acc(te)
    print(f"[train] {name:14s} train acc {a_tr:.3f}  test acc {a_te:.3f}")
    return a_te


def train_all(feats, labels_np, train_idx, test_idx, weights_dir="weights",
              seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    os.makedirs(weights_dir, exist_ok=True)
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    mask = torch.as_tensor(feats["mask"])
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    tr = torch.as_tensor(train_idx); te = torch.as_tensor(test_idx)

    jobs = [
        ("mlp",  MLPNet(),      [T(feats["f_mlp"])],                    200),
        ("drl",  DRLNet(),      [T(feats["f_drl"])],                    200),
        ("lstm", LSTMAttnNet(), [T(feats["f_seq"]), T(feats["f_stat"])], 80),
        ("gnn",  GNNNet(),      [T(feats["f_gnn"])],                    200),
    ]
    for name, model, inputs, epochs in jobs:
        _train(model, inputs, labels, mask, tr, te, epochs, name)
        w = {k: v.detach().numpy().astype(np.float64)
             for k, v in model.export().items()}
        np.savez(os.path.join(weights_dir, name + ".npz"), **w)
    print(f"[train] weights exported -> {weights_dir}/")
