"""taxmeter.measure — the paper's measurement protocol + PT computation.

Protocol (paper §4): 300 sampled events; 50 timed inferences per event with a
20-call warmup (15,000 timed inferences per predictor); memory via
tracemalloc over a 10-call sequence; energy = wall-clock x scenario TDP.

Energy backends: "tdp" (the paper's proxy) and "rapl" (direct Intel RAPL via
/sys/class/powercap — the IMC upgrade path).  RAPL is implemented but
UNTESTED in this container (no powercap here); run it on bare metal.
"""
import csv
import glob
import os
import time
import tracemalloc

import numpy as np

SCENARIOS = {
    "satellite": dict(tdp_w=15.0, e_budget_uj=5000.0, l_budget_ms=100.0,
                      m_budget_mb=64.0, grid_g_kwh=50.0),
    "ground":    dict(tdp_w=150.0, e_budget_uj=500000.0, l_budget_ms=10.0,
                      m_budget_mb=512.0, grid_g_kwh=386.0),
}
DEC_PER_ORBIT = 540            # 90 min orbit / one decision per 10 s
GRID_PRICE_PER_KWH = 0.12
MEM_PRICE_GB_HR = 0.005
HOST_MBPS = 4.3                # host throughput; calibrated to the paper's C scale
GB_PER_DECISION = HOST_MBPS * 1e6 * 10.0 / 8.0 / 1e9


# ------------------------------ accuracy ------------------------------------

def macro_f1(y_true, y_pred, k=5):
    f1s = []
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    for c in range(k):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp == 0 and tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(pred, events, idx):
    y, p = [], []
    for i in idx:
        y.append(events[i].label)
        p.append(pred.predict(events[i]))
    y = np.array(y); p = np.array(p)
    return float((y == p).mean()), macro_f1(y, p)


# --------------------------- energy backends --------------------------------

class TDPBackend:
    name = "tdp"
    def energy_j_per_call(self, mean_call_s, tdp_w):
        return mean_call_s * tdp_w


class RAPLBackend:
    """Direct package-energy measurement via Intel RAPL (UNTESTED here)."""
    name = "rapl"
    PATHS = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))
    def __init__(self):
        if not self.PATHS:
            raise RuntimeError("RAPL not available on this machine")
        self.max_uj = [int(open(p.replace("energy_uj", "max_energy_range_uj"))
                           .read()) for p in self.PATHS]
    def _read(self):
        return [int(open(p).read()) for p in self.PATHS]
    def measure_block(self, fn, n_calls):
        e0 = self._read(); t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        dt = time.perf_counter() - t0
        e1 = self._read()
        d = sum((b - a) % m for a, b, m in zip(e0, e1, self.max_uj))
        return d / 1e6 / n_calls, dt / n_calls   # J/call, s/call


# ------------------------- latency / memory ---------------------------------

def measure_cost(pred, events, n_events=300, warmup=20, timed=50, seed=7):
    rs = np.random.default_rng(seed)
    idx = rs.choice(len(events), size=min(n_events, len(events)),
                    replace=False)
    lat_ns = np.empty(len(idx) * timed)
    k = 0
    pc = time.perf_counter_ns
    for i in idx:
        ev = events[i]
        for _ in range(warmup):
            pred.predict(ev)
        for _ in range(timed):
            t0 = pc()
            pred.predict(ev)
            lat_ns[k] = pc() - t0
            k += 1
    ev0 = events[int(idx[0])]
    pred.predict(ev0)
    tracemalloc.start()
    for _ in range(10):
        pred.predict(ev0)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = (peak + pred.params * 8) / 1e6      # activations + parameters
    lat_ms = lat_ns / 1e6
    # bootstrap 95% CI on the p99.9 (resample the timed calls, 1,000 draws)
    bs = np.random.default_rng(seed + 1)
    boots = np.array([np.percentile(lat_ms[bs.integers(0, k, k)], 99.9)
                      for _ in range(1000)])
    return dict(mean_s=float(lat_ns.mean() / 1e9),
                p50_ms=float(np.percentile(lat_ms, 50)),
                p999_ms=float(np.percentile(lat_ms, 99.9)),
                p999_ci_lo=float(np.percentile(boots, 2.5)),
                p999_ci_hi=float(np.percentile(boots, 97.5)),
                mem_mb=float(mem_mb),
                n_timed=int(k))


# ------------------------------ PT vector -----------------------------------

def pt_axes(cost, scenario, backend=None):
    sc = SCENARIOS[scenario]
    e_j = (backend or TDPBackend()).energy_j_per_call(cost["mean_s"],
                                                      sc["tdp_w"])
    e_uj = e_j * 1e6
    kwh_per_dec = e_j / 3.6e6
    dollar_orbit = (kwh_per_dec * DEC_PER_ORBIT * GRID_PRICE_PER_KWH
                    + cost["mem_mb"] / 1e3 * MEM_PRICE_GB_HR * 1.5)
    carbon_g_gb = kwh_per_dec / GB_PER_DECISION * sc["grid_g_kwh"]
    pt_star = 100.0 * max(e_uj / sc["e_budget_uj"],
                          cost["p999_ms"] / sc["l_budget_ms"],
                          cost["mem_mb"] / sc["m_budget_mb"])
    binding = ["E", "L", "M"][int(np.argmax(
        [e_uj / sc["e_budget_uj"], cost["p999_ms"] / sc["l_budget_ms"],
         cost["mem_mb"] / sc["m_budget_mb"]]))]
    return dict(E_uj=e_uj, M_mb=cost["mem_mb"], L_p999_ms=cost["p999_ms"],
                D_orbit=dollar_orbit, C_g_gb=carbon_g_gb,
                PT_star=pt_star, binding=binding)


def budget_sweep(rows, budgets_mj=(1, 2, 3.5, 5, 7, 8.5, 10, 20, 50, 100)):
    out = []
    for r in rows:
        for b in budgets_mj:
            out.append(dict(predictor=r["predictor"], budget_mj=b,
                            pt_energy_pct=100.0 * r["E_uj_sat"] / (b * 1e3)))
    return out


# ------------------------------ reporting -----------------------------------

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[results] wrote {path} ({len(rows)} rows)")
