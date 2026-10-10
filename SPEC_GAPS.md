# SPEC_GAPS: what a from-spec rebuild of TaxMeter could not pin down

This repo rebuilds the TaxMeter harness *only* from the prose of the
written specification of implementation A. Every item below is a place where the paper's text does not
determine the implementation. Each one is (a) a choice we had to make, and
(b) a sentence a fuller description should add. Items marked **[finding]** are
substantive results of the rebuild, not just bookkeeping.

## 1. Constellation size: 4,408 vs 72x22 = 1,584
The paper claims "4,408-satellite Walker-delta constellation matching FCC
SAT-MOD-20200417-00037" and in the same sentence gives "72 orbital planes at
53.0 deg inclination and 22 satellites per plane" — which is 1,584 satellites
(the Gen-1 53.0 deg shell only). The full license is 5 shells. Rebuild
default: the 1,584-sat shell as described; `--full-gen1` builds all 4,408.
Candidate statistics (and therefore attention cost, which is O(N^2) in
candidates per E2) depend on this.

## 2. No link budget
Events carry RSRP but the paper gives no frequency, EIRP, antenna gains,
path-loss, or fading model. Rebuild: Ku-band 11.7 GHz, EIRP 66.7 dBm,
G_rx 33 dBi (RSRP ~ -69 dBm at zenith), atmospheric 0.5/sin(el) dB, slow
shadowing sigma=2 dB with 30 s coherence, fast scintillation sigma=3 dB iid.
**[finding]** Without the fast-fading term, greedy max-RSRP scores ~0.8+
accuracy — far above the paper's 0.370. The paper's 33-37% band *requires*
substantial irreducible noise in the ground truth; the text never says so.

## 3. Event trigger: 5,969 events out of 7,205 possible ticks
Four hours at one decision per 10 s over five cities = 7,205 ticks, and with
this shell every tick has a serving satellite plus >=2 candidates at 25 deg
min elevation. Our rebuild therefore emits 7,205 events; the paper's 5,969
implies an unstated filter dropping ~17% of ticks (stricter elevation gate,
a handover-warranted condition, or startup exclusion).

## 4. "Acc (F1)" is accuracy, not F1  **[finding]**
Table 1's header conflates the two. Rebuild evidence: 3GPP-A3 and Greedy
predict the argmax-RSRP candidate, i.e. class 0 nearly always after
candidates are sorted by current RSRP. They score accuracy 0.338 but
macro-F1 0.101 in our rebuild — a constant-class predictor *cannot* have
macro-F1 ~= accuracy, yet the paper shows both baselines at 0.370 alongside
the learned models. The reported column is therefore top-1 accuracy. The IMC
version should report both and rename the column.

## 5. Training procedure is absent
"Re-implemented in pure NumPy from its published description" covers
inference only. Nothing states how PHandover-MLP, DHO-DRL, LSTM-Attn, and
GNN-DRL were trained (a DRL environment cannot be re-run from prose).
Rebuild: all four trained *supervised* on the best-target label, 80/20
split, Adam — a documented divergence for the two DRL-derived
architectures. Weights affect only the accuracy column, not the cost axes.

## 6. Feature vectors and architectures are unspecified — but parameter
##    counts constrain them  **[finding]**
The stated counts are strong fingerprints:
- **LSTM-Attn 31,041 reproduces EXACTLY** with LSTM(6->64, combined-bias
  convention: 18,176) + candidate-embed 4->64 (320) + Wq/Wk/Wv 64x64
  (12,480) + scalar combine 64->1 (65). This is very likely the original
  architecture, including the combined-bias parameter-counting convention.
- **PHandover-MLP 9,473 reproduces EXACTLY** as an 81-dim input -> 64 -> 64
  -> 1 scorer (81x64+64 + 64x64+64 + 65 = 9,473). Our rebuild uses a 26-dim
  per-candidate feature set (8,865 params, -6.4%); the 81-dim solution is
  the hypothesis to check against the original code.
- DHO-DRL: 119 -> 128 -> 96 -> 6 gives 28,326 (+4.4% vs 27,137).
- GNN-DRL: two GAT layers 12 -> 64 -> 56 give 4,771 (-3.2% vs 4,929).

## 7. Kalman's information asymmetry is not reproducible from the text
The paper says Kalman "exploits ephemeris information the other predictors
are not given" (52.3% vs the 33-37% band). The feature vectors being
unspecified, our rebuild gives *every* predictor elevation/rate/range —
and Kalman's edge disappears (0.363, inside the band). **[finding]** The
52.3% is a feature-access artifact, not a filtering result; a fuller version
should either equalize inputs or make the asymmetry explicit and justified.

## 8. Walker phasing F is unspecified (rebuild uses F=1).

## 9. Warmup scope is ambiguous
"50 timed inferences per event with a 20-call warmup" — per event or once
per predictor? Rebuild: 20 warmup calls per event (21,000 total calls,
15,000 timed, matching the paper's stated timed total).

## 10. Memory accounting
tracemalloc traces Python-heap allocations only; parameters loaded before
tracing are invisible to it. Rebuild adds params x 8 bytes explicitly. The
paper's "including model parameters" claim needs this (or equivalent) to be
true.

## 11. Dollar and carbon constants
Grid price, memory price, and the host data volume behind "carbon per
gigabyte delivered" are unstated. Back-solving the paper's C scale from its
own E numbers implies ~186 decisions/GB, i.e. ~4.3 Mbps host throughput;
the rebuild uses exactly that and marks it a calibration.

---

# Implementation-sensitivity findings (rebuild vs paper, same spec)

These are the honest headline of the rebuild: which claims survive a
from-spec reimplementation on different hardware, and which do not.

**Survives (qualitative claims):**
- Rules vs learned/stateful predictors differ by 2-3 orders of magnitude in
  per-decision energy and tail latency (289x and 75x here; 1,460x and 238x
  in the paper).
- The binding cost dimension flips from Energy (satellite budgets) to
  Latency (ground budgets) for every predictor — F#2 reproduces exactly.
- 3GPP-A3 and Greedy coincide on target selection (identical accuracy),
  explaining the paper's identical 0.370 entries.
- The learned predictors cluster in a narrow accuracy band (0.32-0.38 here;
  0.33-0.37 in the paper) despite 289x cost spread — F#1's shape reproduces.

**Does not survive (quantitative claims):**
- LSTM-Attn consumes 45% of the 5 mJ satellite budget here, not 150%. Its
  feasibility crossing lands near 2.3 mJ, not the paper's 8.5 mJ. The
  infeasibility verdict at the 5 mJ point is implementation- and
  machine-dependent; F#3's *direction* (selection depends on deployment)
  holds, its *verdict at 5 mJ* does not.
- Mid-field cost ordering flips: our literal per-candidate Kalman loop
  (281 us mean) is *heavier* than LSTM-Attn (150 us), the reverse of the
  paper's Table 1. "Literal NumPy" does not pin down relative cost among
  light and mid predictors.
- The +-4% held-out replication claim fails on a shared vCPU (+-36%
  worst-case here). It plausibly holds on an idle pinned machine — which is
  exactly what the artifact evaluation instructions must require.
