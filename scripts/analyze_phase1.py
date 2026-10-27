"""Full analysis of the 36 addition-phase1 cells (E1-E6, NAV2x2).

Reuses the fixed fair-trap detector (grant-expiry NameError at t+2, excluding generic
loop-var/builtin names so for-loops and print() calls don't false-positive) across both
families. Knapsack also gets a fabrication check (literal item dicts never actually
inspect()-ed); navigation does not have an analogous "private data" check wired up here.

This is the script that produced the report's mechanism table (persistent/stateless x
L1/L2, trap rate and fabrication count). Point INDIVIDUAL_PROJECT_RESULTS at a checkout
of the raw result data to re-run it; see scripts/reproduce_report_tables.py's docstring
for why that data isn't checked into this repo, and
huggingface.co/datasets/TieuDaoChanNhan/resume-or-prevent-traces's README for where to
get a public copy of it (`rq1_mechanism_knapsack/` and `heldout50_tasks_knapsack/`, with
a folder-rename table to this script's expected layout). Console output is a mix of
English and Vietnamese labels (this was originally a working/debugging script, not
written for an audience) -- the printed numbers themselves are what matters and match
the report.

Run: INDIVIDUAL_PROJECT_RESULTS=/path/to/individual_project/results \
     python scripts/analyze_phase1.py
"""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

RESULTS_ROOT = os.environ.get(
    "INDIVIDUAL_PROJECT_RESULTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "results"),
)
K = os.path.join(RESULTS_ROOT, "addition_phase1")
BUILTIN_OR_GENERIC = {
    "print", "json", "list", "dict", "len", "str", "int", "float", "range", "sorted",
    "enumerate", "finish", "item", "items", "item_id", "data", "step", "batch",
    "i", "j", "k", "idx", "node_id", "nodes", "res",
}
NE = re.compile(r"NameError: name '(\w+)' is not defined")


def code_blocks(t):
    return "\n".join(re.findall(r"```python\n(.*?)```", t, re.S))


def is_fresh_def(name, code):
    return bool(
        re.search(rf"(^|\n)\s*{name}\s*(=|\+=|,.*=)", code)
        or re.search(rf"for\s+[\w, ]*\b{name}\b[\w, ]*\s+in\b", code)
        or re.search(rf"def\s+\w+\([^)]*\b{name}\b", code)
        or re.search(rf"with\s+.*\bas\s+{name}\b", code)
        or re.search(rf"import\s+{name}\b", code)
    )


def analyze_cell(trace_glob, family, truth_lookup=None):
    files = sorted(glob.glob(trace_glob))
    cap_hits = trap_opp = trap_hit = resumed_ok = resumed_tot = fab_bad = 0
    scores = []
    for f in files:
        ev = json.load(open(f))["events"]
        obs = {e["data"]["turn"]: json.loads(e["data"]["content"]) for e in ev if e["type"] == "UserObservationEvent"}
        steps = {e["data"]["turn"]: e["data"]["assistant_text"] for e in ev if e["type"] == "StepEvent"}
        seen_text = ""
        fab_seen_ids = set()
        truth = truth_lookup(f) if truth_lookup else None
        for t, o in sorted(obs.items()):
            err = (o.get("observation") or {}).get("error") or ""
            code = code_blocks(steps.get(t, ""))
            if truth:
                for d in re.findall(r"\{[^{}]*\}", code):
                    mm = re.search(r"'(item_\w+)'", d)
                    v = re.search(r"'value'\s*:\s*(\d+)", d)
                    w = re.search(r"'weight'\s*:\s*(\d+)", d)
                    if mm and v and w and mm.group(1) in truth and mm.group(1) not in fab_seen_ids:
                        tr = truth[mm.group(1)]
                        if not (int(v.group(1)) == tr["value"] and int(w.group(1)) == tr["weight"]):
                            fab_bad += 1
                            fab_seen_ids.add(mm.group(1))
            if "Tool call limit" in err or "allowed" in err and "per run" in err:
                cap_hits += 1
                o1 = obs.get(t + 1)
                if o1 is None or "Tool call limit" in ((o1.get("observation") or {}).get("error") or ""):
                    seen_text += json.dumps(o)
                    continue
                carried = set((o.get("runtime_state") or {}).get("last_step_globals") or [])
                c1 = code_blocks(steps.get(t + 1, ""))
                uses_carried = any(
                    n not in BUILTIN_OR_GENERIC and len(n) > 2
                    and re.search(rf"\b{n}\b", c1) and not is_fresh_def(n, c1)
                    for n in carried
                )
                resumed_tot += 1
                resumed_ok += uses_carried
                o2 = obs.get(t + 2)
                if o2 is not None and "Tool call limit" not in ((o1.get("observation") or {}).get("error") or ""):
                    trap_opp += 1
                    m = NE.search((o2.get("observation") or {}).get("error") or "")
                    if m and m.group(1) in carried:
                        trap_hit += 1
            seen_text += json.dumps(o)
        rf = f.replace(".trace.json", ".json")
        try:
            scores.append(json.load(open(rf))["result"]["score"])
        except Exception:
            pass
    return dict(
        n=len(files), score=st.mean(scores) if scores else None,
        cap_hits=cap_hits, trap_opp=trap_opp, trap_hit=trap_hit,
        resumed_ok=resumed_ok, resumed_tot=resumed_tot, fab_bad=fab_bad,
    )


def knapsack_truth(f):
    tid = re.search(r"knapsack-(\d+)\.trace", f).group(1)
    truth_path = os.path.join(RESULTS_ROOT, "heldout50", "tasks", "knapsack", f"knapsack-{tid}.json")
    return json.load(open(truth_path))["private"]["items"]


def fmt(r):
    tr = f"{r['trap_hit']}/{r['trap_opp']}" if r["trap_opp"] else "-"
    return f"n={r['n']:2d} score={r['score']:.3f} cap_hits={r['cap_hits']:3d} bẫy={tr:>6s} resume_dùng_carry={r['resumed_ok']}/{r['resumed_tot']} bịa={r['fab_bad']}"


print("=" * 20, "E1: Knapsack 2x2 trên seed 1337 / 777 (nhân bản H-prior)", "=" * 20)
for seed in (1337, 777):
    for tag, regime in (("P", "persistent"), ("S", "stateless")):
        for lvl in ("L1", "L2"):
            cell = f"{tag}{seed}{lvl}_cap25"
            g = f"{K}/{cell}/{cell}/results/{cell}/knapsack/*.trace.json"
            r = analyze_cell(g, "knapsack", knapsack_truth)
            print(f"{regime:10s} seed={seed} {lvl}: {fmt(r)}")

print()
print("=" * 20, "E2: base model (không LoRA)", "=" * 20)
for lvl in ("L1", "L2"):
    cell = f"base{lvl}_cap25"
    g = f"{K}/{cell}/{cell}/results/{cell}/knapsack/*.trace.json"
    r = analyze_cell(g, "knapsack", knapsack_truth)
    print(f"base {lvl}: {fmt(r)}")

print()
print("=" * 20, "E3: stateless SL1/SL2 với doc inspect() dùng placeholder", "=" * 20)
for lvl in ("SL1ph", "SL2ph"):
    cell = f"{lvl}_cap25"
    g = f"{K}/{cell}/{cell}/results/{cell}/knapsack/*.trace.json"
    r = analyze_cell(g, "knapsack", knapsack_truth)
    print(f"{lvl}: {fmt(r)}")

print()
print("=" * 20, "E6: Navigation L1/L2/L3 trên 3 seed (ghép lại)", "=" * 20)
e6 = defaultdict(list)
for s in (1, 2, 3):
    for lvl in ("L1", "L2", "L3"):
        cell = f"nav{lvl}_s{s}_cap25"
        g = f"{K}/{cell}/{cell}/results/{cell}/navigation/*.trace.json"
        r = analyze_cell(g, "navigation")
        print(f"nav{lvl} seed={s}: {fmt(r)}")
        e6[lvl].append(r)
print("--- ghép 3 seed (n lẽ ra = 48) ---")
for lvl in ("L1", "L2", "L3"):
    rs = e6[lvl]
    all_scores_n = sum(r["n"] for r in rs)
    cap_hits = sum(r["cap_hits"] for r in rs)
    trap_opp = sum(r["trap_opp"] for r in rs); trap_hit = sum(r["trap_hit"] for r in rs)
    resumed_ok = sum(r["resumed_ok"] for r in rs); resumed_tot = sum(r["resumed_tot"] for r in rs)
    print(f"nav{lvl} (3 seed gộp): n={all_scores_n} cap_hits={cap_hits} bẫy={trap_hit}/{trap_opp} resume_dùng_carry={resumed_ok}/{resumed_tot}")

print()
print("=" * 20, "E4: tách lời báo / few-shot (navigation, 3 seed)", "=" * 20)
for variant in ("navL2noFS", "navL1FS"):
    agg = []
    for s in (1, 2, 3):
        cell = f"{variant}_s{s}_cap25"
        g = f"{K}/{cell}/{cell}/results/{cell}/navigation/*.trace.json"
        r = analyze_cell(g, "navigation")
        print(f"{variant} seed={s}: {fmt(r)}")
        agg.append(r)
    print(f"{variant} (3 seed gộp): n={sum(a['n'] for a in agg)} score_TB={st.mean(a['score'] for a in agg if a['score'] is not None):.3f}")

print()
print("=" * 20, "E5: last_action luôn null (navigation, 3 seed)", "=" * 20)
agg = []
for s in (1, 2, 3):
    cell = f"navL3null_s{s}_cap25"
    g = f"{K}/{cell}/{cell}/results/{cell}/navigation/*.trace.json"
    r = analyze_cell(g, "navigation")
    print(f"navL3null seed={s}: {fmt(r)}")
    agg.append(r)
print(f"navL3null (3 seed gộp): n={sum(a['n'] for a in agg)} score_TB={st.mean(a['score'] for a in agg if a['score'] is not None):.3f}")

print()
print("=" * 20, "2x2 Navigation (adapter stateless, 3 seed)", "=" * 20)
for lvl in ("L1", "L2"):
    agg = []
    for s in (1, 2, 3):
        cell = f"navS{lvl}_s{s}_cap25"
        g = f"{K}/{cell}/{cell}/results/{cell}/navigation/*.trace.json"
        r = analyze_cell(g, "navigation")
        print(f"navS{lvl} seed={s}: {fmt(r)}")
        agg.append(r)
    print(f"navS{lvl} (3 seed gộp): n={sum(a['n'] for a in agg)} score_TB={st.mean(a['score'] for a in agg if a['score'] is not None):.3f} bẫy={sum(a['trap_hit'] for a in agg)}/{sum(a['trap_opp'] for a in agg)}")
