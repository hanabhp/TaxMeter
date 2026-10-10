# Use of generative AI in this work

This note accompanies the paper "The Predictability Tax of LEO Handover Predictors" and
follows the ACM Policy on Authorship, which asks authors to disclose generative AI used to
create content in proportion to how much it produced, and to describe AI used in the
research itself.

Tool: Anthropic's Claude, used through the claude.ai interface with code execution, in
September and October 2026.

What the AI system produced, under the author's direction:
- Implementation B of the harness (everything in taxmeter/), written from the paper's
  written specification without access to implementation A's code.
- The experiments run on implementation B and the scripts behind them: the synthetic and
  live-constellation runs, the twenty-city run, operation counts and candidate scaling,
  the bootstrap intervals, the vectorized Kalman variant, batching, the selection map and
  the cascade (experiments/), and the Apple Silicon measurement script. The Apple Silicon
  measurement itself was run by the author on the author's own machine.
- The figures (TikZ and PGFPlots source in the paper) and drafts and edits of the text.
- Checks of the bibliography against Crossref, the arXiv API and issuing sources.

How the output was checked:
- Every number reported in the paper traces to a run file listed in results/PROVENANCE.md.
- Batched and vectorized predictors were checked for decision agreement with the
  one-at-a-time predictors before any timing was reported.
- Two wording errors and several numeric errors found by audits against the run files were
  corrected before submission; the build notes record them.

The author specified the study, directed every step, and takes full responsibility for the
content of the paper and this artifact.
