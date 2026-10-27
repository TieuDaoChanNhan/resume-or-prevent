"""Reproduce the GStar individual-project report's paired-bootstrap tables from raw
per-task result files.

Mirrors scripts/rollout_variance.py and scripts/analyze_paper.py's methodology: paired
bootstrap over task ids shared by both conditions, resampling the SAME task index into
both sides of a draw (never independent resampling of each side). The report calls a
comparison significant when the resulting 95% interval (10,000 resamples) excludes zero.

The raw per-task result files this script reads (tens of thousands of trace/result JSON
files) are not checked into this repo -- they are large and specific to the cluster they
were produced on, the same reason the eval configs under configs/ redact their absolute
paths. This script is what actually computed every paired-bootstrap point estimate, CI,
and significance call in the report; point estimates reproduce exactly, and CI bounds
reproduce within Monte Carlo noise of a few thousandths (the sig/ns call itself always
matches). Point RESULTS_ROOT and INDIVIDUAL_PROJECT_DATA at a checkout of that raw data
to re-run it yourself.

That raw data is publicly released as huggingface.co/datasets/TieuDaoChanNhan/resume-or-prevent-traces,
whose directory names differ from the ones this script reads (renamed for public clarity).
See that dataset's README, "Reproducing the report's numbers from this release", for the
exact folder-rename table -- download the release, rename the folders as shown there, and
point RESULTS_ROOT/INDIVIDUAL_PROJECT_DATA at the renamed copies.

Run: INDIVIDUAL_PROJECT_RESULTS=/path/to/individual_project/results \
     INDIVIDUAL_PROJECT_DATA=/path/to/individual_project_phase2 \
     python scripts/reproduce_report_tables.py
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics

DATA_ROOT = os.environ.get(
    "INDIVIDUAL_PROJECT_DATA",
    "/e/data1/datasets/playground/mmlaion/shared/nguyen38/individual_project_phase2",
)
RESULTS_ROOT = os.environ.get(
    "INDIVIDUAL_PROJECT_RESULTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "results"),
)
NB = 10_000


def _scores(cell_dir: str, family: str) -> dict[str, float]:
    """task id (the zero-padded numeric suffix) -> score, for one cell directory."""
    out = {}
    pattern = f"{cell_dir}/**/{family}-{family}-*.json"
    for f in sorted(glob.glob(pattern, recursive=True)):
        if f.endswith(".trace.json"):
            continue
        d = json.load(open(f)).get("result", {}) or {}
        v = d.get("score")
        if v is None:
            continue
        tid = os.path.basename(f).rsplit("-", 1)[-1].split(".")[0]
        out[tid] = float(v)
    return out


def paired_bootstrap(dir_a: str, dir_b: str, family: str = "knapsack",
                      resamples: int = NB, seed: int = 0):
    """Paired bootstrap of mean(b) - mean(a) over the task ids present in both cells."""
    a = _scores(dir_a, family)
    b = _scores(dir_b, family)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(f"no shared task ids between {dir_a} and {dir_b}")
    va = [a[t] for t in common]
    vb = [b[t] for t in common]
    diffs = [y - x for x, y in zip(va, vb)]
    n = len(diffs)
    point = sum(diffs) / n

    rng = random.Random(seed)
    boot = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    lo = boot[int(0.025 * resamples)]
    hi = boot[int(0.975 * resamples)]
    sig = lo > 0 or hi < 0
    return {
        "n": n,
        "mean_a": statistics.mean(va),
        "mean_b": statistics.mean(vb),
        "diff": point,
        "ci": (lo, hi),
        "sig": sig,
    }


def _fmt(r):
    tag = "sig." if r["sig"] else "ns"
    return (f"n={r['n']:>3}  a={r['mean_a']:.3f} b={r['mean_b']:.3f} "
            f"diff={r['diff']:+.3f}  CI=[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  {tag}")


def _p(base, *parts):
    return os.path.join(base, *parts)


def table_ladder_pairs():
    """tab:ladder-scores / tab:ladder-pairs: L0-L3 on the main 50-instance pool."""
    base = _p(RESULTS_ROOT, "heldout50_ladder")
    levels = {lv: _p(base, f"{lv}_cap25") for lv in ("L0", "L1", "L2", "L3")}
    print("=== tab:ladder-pairs (results/heldout50_ladder) ===")
    for label, (x, y) in {
        "L1-L0": ("L0", "L1"), "L2-L1": ("L1", "L2"), "L3-L2": ("L2", "L3"),
        "L3-L1": ("L1", "L3"), "L3-L0": ("L0", "L3"),
    }.items():
        r = paired_bootstrap(levels[x], levels[y])
        print(f"  {label:8s} {_fmt(r)}")


def table_wording():
    """tab:wording: the L3-L2 rerun + 3 wording variants, including the n=46 few-shot cell."""
    print("=== tab:wording (results/heldout50_ladder{,_rerun,_a,_c,_fewshot}) ===")
    rows = {
        "Original": (_p(RESULTS_ROOT, "heldout50_ladder", "L2_cap25"),
                     _p(RESULTS_ROOT, "heldout50_ladder", "L3_cap25")),
        "Original, rerun": (_p(RESULTS_ROOT, "heldout50_ladder_rerun", "L2rerun_cap25"),
                             _p(RESULTS_ROOT, "heldout50_ladder_rerun", "L3rerun_cap25")),
        "Added rule (note)": (_p(RESULTS_ROOT, "heldout50_ladder_a", "L2note_cap25"),
                               _p(RESULTS_ROOT, "heldout50_ladder_a", "L3note_cap25")),
        "Shown when relevant (inline)": (_p(RESULTS_ROOT, "heldout50_ladder_c", "L2inline_cap25"),
                                          _p(RESULTS_ROOT, "heldout50_ladder_c", "L3inline_cap25")),
        "Worked example (few-shot), n=46": (_p(RESULTS_ROOT, "heldout50_ladder_fewshot", "L2fs_cap25"),
                                             _p(RESULTS_ROOT, "heldout50_ladder_fewshot", "L3fs_cap25")),
    }
    for label, (x, y) in rows.items():
        r = paired_bootstrap(x, y)
        print(f"  {label:32s} {_fmt(r)}")


def table_nav_rep50():
    """tab:nav-rep50: Navigation L0-L3 cross-domain check, 50 fresh instances."""
    base = _p(RESULTS_ROOT, "nav_heldout50_ladder")
    levels = {lv: _p(base, f"nav{lv}50_cap25") for lv in ("L0", "L1", "L2", "L3")}
    print("=== tab:nav-rep50 (results/nav_heldout50_ladder) ===")
    for label, (x, y) in {
        "L1-L0": ("L0", "L1"), "L2-L0": ("L0", "L2"), "L3-L0": ("L0", "L3"),
        "L2-L1": ("L1", "L2"), "L3-L2": ("L2", "L3"),
    }.items():
        r = paired_bootstrap(levels[x], levels[y], family="navigation")
        print(f"  {label:8s} {_fmt(r)}")


def table_nav_ablation():
    """The two targeted Navigation ablations, same 50-instance pool."""
    base = _p(RESULTS_ROOT, "nav_heldout50_ladder")
    print("=== Navigation ablations (results/nav_heldout50_ladder) ===")
    pairs = {
        "(L2 ex. removed)-L2": (_p(base, "navL250_cap25"), _p(base, "navL2noFS50_cap25")),
        "(L1 ex. added)-L1": (_p(base, "navL150_cap25"), _p(base, "navL1FS50_cap25")),
        "(L1 ex. added)-(L2 ex. removed)": (_p(base, "navL2noFS50_cap25"), _p(base, "navL1FS50_cap25")),
        "L3-(L3 last_action blanked)": (_p(base, "navL3null50_cap25"), _p(base, "navL350_cap25")),
    }
    for label, (x, y) in pairs.items():
        r = paired_bootstrap(x, y, family="navigation")
        print(f"  {label:34s} {_fmt(r)}")


def table_rq2_main():
    """tab:rq2-main: prevention vs. recovery, n=56 and n=103, cap 25 and cap 80."""
    print("=== tab:rq2-main (individual_project_phase2/eval, eval_180) ===")
    scales = {"n=56": _p(DATA_ROOT, "eval"), "n=103": _p(DATA_ROOT, "eval_180")}
    for scale, base in scales.items():
        for cap in (25, 80):
            prev = _p(base, f"PREV_cap{cap}")
            rec = _p(base, f"REC_cap{cap}")
            r = paired_bootstrap(rec, prev)  # diff reported is Prev - Rec
            print(f"  {scale} cap{cap}: {_fmt(r)}")


def table_rq2_rerun():
    """The RQ2 plain-rerun check on the n=103 adapters."""
    print("=== RQ2 rerun (individual_project_phase2/eval_180_rerun) ===")
    base = _p(DATA_ROOT, "eval_180_rerun")
    for cap in (25, 80):
        prev = _p(base, f"PREVrerun_cap{cap}")
        rec = _p(base, f"RECrerun_cap{cap}")
        r = paired_bootstrap(rec, prev)
        print(f"  cap{cap}: {_fmt(r)}")


if __name__ == "__main__":
    table_ladder_pairs()
    table_wording()
    table_nav_rep50()
    table_nav_ablation()
    table_rq2_main()
    table_rq2_rerun()
