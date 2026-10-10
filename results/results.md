# TaxMeter rebuild — dev-box results (development-grade timing)

1,584-sat Gen-1 shell 1 (as the paper's text describes), 7,205 events over
4 h x 5 cities, seeds fixed. Shared 1-vCPU container: latency/energy are
DEVELOPMENT-GRADE (see README); accuracy and parameter counts are final.

| Predictor | params (stated / rebuilt) | acc | macro-F1 | mean us | p99.9 ms | E uJ (sat) | PT*_sat | bind sat | PT*_gnd | bind gnd |
|---|---|---|---|---|---|---|---|---|---|---|
| 3GPP-A3 | 0 / 0 | 0.338 | 0.101 | 2.4 | 0.028 | 36 | 0.7% | E | 0.3% | L |
| Greedy | 0 / 0 | 0.338 | 0.101 | 1.0 | 0.017 | 15 | 0.3% | E | 0.2% | L |
| Kalman | 0 / 0 | 0.363 | 0.328 | 281.4 | 1.283 | 4,221 | 84.4% | E | 12.8% | L |
| PreHO-Geo | 0 / 0 | 0.363 | 0.329 | 14.9 | 0.078 | 223 | 4.5% | E | 0.8% | L |
| PHandover-MLP | 9,473 / 8,865 | 0.380 | 0.299 | 14.9 | 0.061 | 224 | 4.5% | E | 0.6% | L |
| DHO-DRL | 27,137 / 28,326 | 0.362 | 0.270 | 10.0 | 0.042 | 149 | 3.0% | E | 0.4% | L |
| LSTM-Attn | 31,041 / **31,041** | 0.384 | 0.282 | 150.0 | 0.973 | 2,251 | 45.0% | E | 9.7% | L |
| GNN-DRL | 4,929 / 4,771 | 0.317 | 0.135 | 57.9 | 0.346 | 868 | 17.4% | E | 3.5% | L |

Headline spreads: 289x energy, 75x p99.9 (paper: 1,460x, 238x).

Reproduces: narrow accuracy band + orders-of-magnitude cost spread (F#1's
shape); the Energy->Latency binding flip between satellite and ground for
every predictor (F#2, exactly); A3 == Greedy on selection. Does not
reproduce: LSTM-Attn infeasibility at 5 mJ (45% here, crossing ~2.3 mJ vs
the paper's 8.5 mJ); mid-field cost order (Kalman heavier than LSTM-Attn
here); the +-4% held-out replication on shared hardware (+-36% worst case).
Full analysis in SPEC_GAPS.md.
