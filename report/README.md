# Resume or Prevent?

An individual project for the GStar Bootcamp 2026, expanding on the runtime-contracts study
described in the repository's top-level [README](../README.md). This directory is **not part of
the original paper**: it is a separate study by Van Khue Nguyen, built on top of that paper's
released code, data, and trained adapters under their Apache-2.0 license. The paper's own
authors are not responsible for the content of this directory.

The full write-up, including methodology, every results table, and limitations, is
[`GStar2026_Individual_Project_Report_Van_Khue_Nguyen.pdf`](GStar2026_Individual_Project_Report_Van_Khue_Nguyen.pdf)
(source: the accompanying `.tex` file). This README is an index, not a substitute for it.

## Scope

The paper leaves two questions open: whether an agent also needs to be told *where* its
truncated work stopped and what to do next (beyond just which variables survived), and whether
*retraining* an agent to handle a tool-call cap differently closes the quality gap the paper
reports, rather than only changing what a fixed agent is told at inference time. This project
answers both, on the paper's own Opaque Knapsack task plus a second task family, Navigation:

- **RQ1** (resumption metadata vs. announced carryover): tested and found **not supported** —
  an apparent win does not survive a robustness check across wording variants and a plain rerun,
  and is explained by a mechanism (a training-time prior, not a prompt-content effect) rather
  than left unexplained.
- **RQ2** (training a prevention-style vs. a recovery-style strategy): tested by training two new
  QLoRA adapters and found **supported** at a tight tool-call cap, **null** at a slack one,
  matching the direction predicted before training.

## Where the code for this project lives

The configs, benchmark cells, and training recipes this project adds on top of the paper's own
code are under `experiments/cap_sweep/**/addition_phase1/`, `experiments/cap_sweep/**/nav_heldout50/`,
`experiments/cap_sweep/**/heldout50_ladder/` and `heldout50_ladder_rerun/` (Knapsack and Navigation
cells for RQ1 and the mechanism check), and `configs/phase2_eval*/` plus
`train/configs/axolotl_qwen3_8b_{prevention,recovery}*.yaml` (RQ2 training and evaluation). These
sit alongside, and reuse, the paper's own harness and task generators described in the top-level
README; nothing in the harness itself was modified.

## Reproducing the report's numbers

Every number in the report traces back to a committed script run against the raw per-task
result files:

- `scripts/reproduce_report_tables.py` -- every paired-bootstrap point estimate, CI, and
  significance call: the RQ1 ladder and wording tables, the Navigation cross-domain and
  ablation tables, and both RQ2 scales (n=56 and n=103, both caps), using the same
  paired-bootstrap methodology (10,000 resamples) the report describes.
- `scripts/analyze_phase1.py` -- the mechanism table (trap rate and training-trace
  fabrication, persistent vs. stateless x L1/L2).
- `scripts/reproduce_fabrication_table.py` -- the RQ2 test-time fabrication table
  (prevention vs. recovery, both caps), which also cross-checks that the held-out task
  pool it reads ground truth from is really the one the eval traces were generated
  against.
- `train/prepare_paired_data.py`'s manifest reproduces the training-pair yield numbers.

The raw result files themselves are not in this repo (large, cluster-path-specific), so
these scripts need to be pointed at a copy of that data to actually run; see each
script's own docstring.

## Data and checkpoints

This project trains one new pair of adapters: **prevention** and **recovery**, at two scales
(56 and 103 training pairs). Those four checkpoints, plus every raw trace this project
generated (the Knapsack rerun, the Navigation n=50 replication and its ablations, and the RQ2
training/evaluation traces), are released on Hugging Face under the `TieuDaoChanNhan` account,
grouped under one collection.

The mechanistic 2x2 in the "Mechanism" section of the report reuses the paper's own
already-released Knapsack and Navigation persistent/stateless adapters directly (no new
training) — see the top-level README's **Released artifacts** section for those; they are not
duplicated here.

## License

Apache License 2.0, same as the rest of this repository.
