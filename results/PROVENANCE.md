# Which released run file backs which number in the paper

Development-grade timing drifts between runs on the shared host (the paper reports up to 36%),
so every row and point in the paper is tied to one named file here.

| Paper element | File | Notes |
|---|---|---|
| Table 1, column B, rows 3GPP-A3, Greedy, Kalman, PreHO-Geo, PHandover-MLP, DHO-DRL, LSTM-Attn | table1_rebuild.csv (run of 2026-09-30) | first complete B run; text numbers 289x, 45%, 2.3 mJ, 281 us, 150 us come from it |
| Table 1, column B, row GNN-DRL | table1_devgrade_rev5.csv (run of 2026-10-06) | GNN redesigned (GATv2 with root term, 4,853 params) after a degenerate first layer was caught; re-measured with the same protocol |
| Table 1, column A (all rows) | quoted from the earlier draft's Table 1 | not produced by this repository |
| Figure 1 and Figure 3 B points | same as the Table 1 B rows above | GNN point from table1_devgrade_rev5.csv |
| Figure 1 live points (5 cities) | table1_devgrade_live.csv (2026-10-03), GNN from table1_devgrade_live_rev5.csv | live weights in weights_live/ |
| Section 5 vectorized Kalman 99 us | review_additions_devgrade.csv | |
| Section 5 bootstrap CI span 5 to 25% | review_additions_devgrade.csv | p999_ci_lo / p999_ci_hi |
| Section 5 twenty-city accuracies and visible counts | live20_per_city.csv | weights_live/ evaluated on workload_live20.npz |
| Section 5 MAC counts and ns per MAC | op_counts_devgrade.csv | counts from weight shapes |
| Section 5 candidate scaling 60x / 3.9x / 1.3x | candidate_scaling_devgrade.csv | whole-predictor cost, K = 5 to 75 |
| Section 4 PyTorch cross-check 1.5 to 3.8x, full decision agreement | runtime_check_devgrade.csv | |
| Section 5 live run energy spread 426x, LSTM-Attn 46% | table1_devgrade_live.csv | |
| Section 5 measured energy (3.6 to 904 uJ, 249x, rank 0.98 and 0.93, 10% of the satellite budget, 3.9 to 5.4 W marginal, 10 to 18% between repetitions) and Section 6 factor 3.1 to 3.5 | mac_energy_runs.csv, mac_energy_summary.csv, mac_energy_console.txt | Apple Silicon MacBook Pro, macOS 15.7.4, powermetrics cpu_power, 2026-10-06; CSVs rebuilt from the console log |
| Section 5 code-versus-host attribution (rank correlation 0.93 for B on two hosts, 0.57 for A against B and against the Mac run) | computed from results/paper_table1.csv (A quoted, B) and results/mac_energy_summary.csv | Spearman over the eight predictors in Table 1 |
| Section 5 batching (13-fold and 37-fold for LSTM-Attn at batches 32 and 512, comparator up to 183-fold, spread 109x, 225x, 686x, 6% of the satellite budget at batch 32, graph predictor overtaking at large batches, full decision agreement) | batching_devgrade.csv | experiments/batching.py; PyTorch path for the learned predictors, vectorized NumPy for the rules; single thread |
| Figure 5 and the 'Where each predictor wins' paragraph (regions, 157 uJ, 2 mJ, 2.3 mJ, 1 ms, 2.7 mJ margin, A's 7.5 mJ boundary) | selection_inputs_live.csv, selection_map_live.csv | costs from table1_devgrade_live.csv with the GNN row from table1_devgrade_live_rev5.csv; accuracies are the twenty-city means from live20_per_city.csv; A's 7,499 uJ from the quoted Table 1 |
| Section 5 cascade and the second contribution (geometric rule alone 70% of the gain at 7% of LSTM-Attn energy; cascade escalating 23% of decisions at 0.25 dB recovers 81% at 30%; escalating on the current RSRP gap is roughly proportional) | cascade_geo_live20.csv, cascade_live20.csv | experiments/cascade.py on workload_live20.npz with weights_live; energies from table1_devgrade_live.csv at 15 W |
