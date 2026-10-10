"""Smoke tests for the TaxMeter harness. Run with: python -m pytest tests/
or: python tests/test_taxmeter.py"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from taxmeter import core, measure                         # noqa: E402
from taxmeter.predictors import (build_events, featurize_all,  # noqa: E402
                                 load_predictors, Greedy, GPP_A3)


def _workload():
    return dict(np.load(os.path.join(ROOT, "data", "workload.npz")))


def test_parameter_counts_match_paper_table():
    preds = load_predictors(os.path.join(ROOT, "weights"))
    counts = {p.name: p.params for p in preds}
    assert counts["LSTM-Attn"] == 31041        # exact match to the stated count
    assert counts["PHandover-MLP"] == 8865
    assert counts["DHO-DRL"] == 28326
    assert counts["GNN-DRL"] == 4853
    for rule in ("3GPP-A3", "Greedy", "Kalman", "Kalman-vec", "PreHO-Geo"):
        assert counts[rule] == 0


def test_workload_shapes_and_labels():
    d = _workload()
    n = len(d["label"])
    assert n == 7205
    assert d["c_rh"].shape == (n, 5, 8) and d["c_eh"].shape == (n, 5, 8)
    assert d["c_mask"].sum(axis=1).min() >= 2
    assert set(np.unique(d["label"])) <= {0, 1, 2, 3, 4}
    f = featurize_all(d)
    assert f["f_mlp"].shape == (n, 5, 26) and f["f_drl"].shape == (n, 119)
    assert f["f_seq"].shape == (n, 8, 6) and f["f_gnn"].shape == (n, 6, 12)


def test_a3_and_greedy_choose_the_same_target():
    d = _workload()
    events, _ = build_events(d)
    a3, gr = GPP_A3(), Greedy()
    agree = sum(a3.predict(e) == gr.predict(e) for e in events[:2000])
    assert agree == 2000


def test_measurement_protocol_counts():
    d = _workload()
    events, _ = build_events(d)
    cost = measure.measure_cost(Greedy(), events, n_events=20, warmup=2, timed=5)
    assert cost["n_timed"] == 100
    assert cost["p999_ci_lo"] <= cost["p999_ms"] <= cost["p999_ci_hi"] * 1.0001
    ax = measure.pt_axes(cost, "satellite")
    assert ax["binding"] in ("E", "L", "M") and ax["PT_star"] > 0


def test_live_tle_loads_and_propagates():
    sats = core.load_tle_constellation(os.path.join(ROOT, "data", "starlink_celestrak.tle"))
    assert len(sats) == 10679
    d = dict(np.load(os.path.join(ROOT, "data", "workload_live.npz")))
    assert len(d["label"]) == 4800
    # decisions sit on Starlink's 15 s slots at :12/:27/:42/:57
    assert set(np.unique(d["tick"] % 60)) <= {12, 27, 42, 57}


def test_link_budget_anchor():
    assert abs(core.rsrp_deterministic(90.0, 550.0) - (-69.4)) < 0.2


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        if name == "test_live_tle_loads_and_propagates" and core.Satrec is None:
            print("SKIP", name, "(sgp4 is not installed; the energy measurement does not need it)")
            continue
        try:
            fn()
            print("PASS", name)
        except Exception as exc:          # report every check instead of stopping at the first
            failed += 1
            print("FAIL", name, "-", type(exc).__name__, exc)
    sys.exit(1 if failed else 0)
