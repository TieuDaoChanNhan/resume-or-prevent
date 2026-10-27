"""Check the 4 Phase 2 pilot configs' real system prompts before spending any teacher
API budget. Run from runtime-contract-phase2/."""
import json
import sys
from unittest.mock import MagicMock

import yaml

from codeact_runtime.benchmark.config import FAMILY_ENVS, BenchmarkConfig
from codeact_runtime.codeact.agent import (
    STRATEGY_PROMPT_PREVENTION,
    STRATEGY_PROMPT_RECOVERY,
    CodeAct,
)

TASK = "../results/phase2_trainpool/tasks/knapsack/knapsack-0000000000.json"


def build(cfg_path):
    cfg = BenchmarkConfig(**yaml.safe_load(open(cfg_path)))
    (a,), fam = cfg.agents, cfg.task_families[0]
    env = FAMILY_ENVS[fam].from_task(json.load(open(TASK)))
    ag = CodeAct(
        llm_proxy=MagicMock(), max_num_turns=a.max_turns, tools=env.get_tools(), bus=MagicMock(),
        persistent_state=a.persistent_state, max_tool_calls=a.max_tool_calls,
        state_carryover_on_cap=a.state_carryover_on_cap, announce_carryover=a.announce_carryover,
        resumption_metadata=a.resumption_metadata, task_family=fam,
        strategy_prompt=a.strategy_prompt, strategy_block_calls=a.strategy_block_calls,
    )
    return [m["content"] for m in ag.message_history if m["role"] == "system"], a


bad = 0
cells = {
    "PREV_L0": "configs/phase2_pilot/_PREV_L0_cap25.yaml",
    "PREV_L3": "configs/phase2_pilot/_PREV_L3_cap25.yaml",
    "REC_L0": "configs/phase2_pilot/_REC_L0_cap25.yaml",
    "REC_L3": "configs/phase2_pilot/_REC_L3_cap25.yaml",
}
sysmsgs = {}
for name, path in cells.items():
    msgs, a = build(path)
    sysmsgs[name] = msgs
    strat = msgs[-2]  # last system msg before the tool prompt
    tool_prompt = msgs[-1]
    checks = {}
    if name.startswith("PREV"):
        checks["is prevention text"] = "PREVENT TRUNCATION" in strat
        checks["block budget filled in"] = "20 tool calls" in strat
        checks["{max_calls_below_cap} not left unformatted"] = "{max_calls_below_cap}" not in strat
    else:
        checks["is recovery text, byte-identical to constant"] = strat == STRATEGY_PROMPT_RECOVERY
        checks["no runtime name leaked into the text"] = "L0" not in strat and "L3" not in strat
    checks["cap echoed in tool prompt"] = "25 per run" in tool_prompt or "STRICT LIMIT: Do not exceed 25" in tool_prompt
    ok = all(checks.values())
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name:8s} ({len(msgs)} system msgs): " + "; ".join(f"{k}={v}" for k, v in checks.items()))

# cross-cell checks
print()
l0_prev, l3_prev = sysmsgs["PREV_L0"], sysmsgs["PREV_L3"]
same_strategy_text = l0_prev[-2] == l3_prev[-2]
print(f"{'OK ' if same_strategy_text else 'BAD'} PREV strategy text identical across L0/L3: {same_strategy_text}")
bad += not same_strategy_text

l0_rec, l3_rec = sysmsgs["REC_L0"], sysmsgs["REC_L3"]
same_rec_text = l0_rec[-2] == l3_rec[-2] == STRATEGY_PROMPT_RECOVERY
print(f"{'OK ' if same_rec_text else 'BAD'} REC strategy text identical across L0/L3 and matches the constant: {same_rec_text}")
bad += not same_rec_text

l3_has_more = len(l3_prev) > len(l0_prev)  # L3 adds the carryover few-shot example
print(f"{'OK ' if l3_has_more else 'BAD'} L3 cells carry the extra carryover few-shot block vs L0: {l3_has_more}")
bad += not l3_has_more

print("\nALL OK" if not bad else f"\n{bad} MISMATCHES")
sys.exit(1 if bad else 0)
