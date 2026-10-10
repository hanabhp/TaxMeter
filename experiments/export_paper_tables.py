#!/usr/bin/env python3
"""Write the numbers behind the paper's Table 1 and Figures 1, 3 and 4 as one CSV.

Implementation A values are quoted from the earlier draft's Table 1 (not produced by
this repository). Implementation B values come from results/table1_devgrade.csv
(or table1_rebuild.csv for the run used in the submission)."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = {  # name: (params_stated, acc, E_uj_sat, p999_ms, PT_sat, PT_gnd)
    "3GPP-A3":       (0,     0.370, 5,    0.005, 0.10,  0.05),
    "Greedy":        (0,     0.370, 12,   0.018, 0.24,  0.18),
    "Kalman":        (0,     0.523, 490,  0.115, 9.8,   1.15),
    "PreHO-Geo":     (0,     0.337, 177,  0.079, 3.5,   0.79),
    "PHandover-MLP": (9473,  0.353, 807,  0.172, 16.1,  1.72),
    "GNN-DRL":       (4929,  0.370, 1513, 0.293, 30.3,  3.03),
    "DHO-DRL":       (27137, 0.293, 5775, 0.824, 115.5, 11.55),
    "LSTM-Attn":     (31041, 0.370, 7499, 1.106, 150.0, 15.0),
}
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "table1_rebuild.csv")
B = {r["predictor"]: r for r in csv.DictReader(open(src))}
# the GNN row of the paper comes from the post-redesign run (see results/PROVENANCE.md)
rev5 = os.path.join(ROOT, "results", "table1_devgrade_rev5.csv")
if os.path.exists(rev5):
    for r in csv.DictReader(open(rev5)):
        if r["predictor"] == "GNN-DRL":
            B["GNN-DRL"] = r
out = os.path.join(ROOT, "results", "paper_table1.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["predictor", "params_stated", "params_B", "acc_A", "acc_B", "E_uj_A", "E_uj_B",
                "p999_ms_A", "p999_ms_B", "PT_sat_A", "PT_sat_B", "PT_gnd_A", "PT_gnd_B"])
    for name, (ps, acc, e, l, pts, ptg) in A.items():
        b = B[name]
        w.writerow([name, ps, b["params_rebuilt"], acc, b["acc"], e, b["E_uj_sat"], l,
                    b["p999_ms_devbox"], pts, b["PT_star_sat"], ptg, b["PT_star_gnd"]])
print("wrote", out)
