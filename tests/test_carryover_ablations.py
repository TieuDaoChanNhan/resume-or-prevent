"""Ablations that pull apart what the announced carryover variants bundle.

Each flag must change exactly one thing, and leaving it unset must leave the existing
cells byte-identical: the L1/L2/L3 runs already on disk are the comparison points.
"""

import json

import pytest

from codeact_runtime.benchmark.benchmark import use_placeholder_tool_examples
from codeact_runtime.codeact.agent import (
    CARRYOVER_FEWSHOT_EXAMPLES,
    STATELESS_FEWSHOT_EXAMPLES,
    CodeAct,
)
from codeact_runtime.codeact.events import EventBus
from codeact_runtime.families.knapsack import (
    INSPECT_DOC_EXAMPLE,
    INSPECT_DOC_EXAMPLE_PLACEHOLDER,
    InspectTool,
    KnapsackEnv,
)

from tests.test_cap_carryover import CAP, REPLIES, _Ping, _ScriptedProxy

L2_EXAMPLE, L3_EXAMPLE = CARRYOVER_FEWSHOT_EXAMPLES["knapsack"]


def _agent(replies=REPLIES, **kwargs) -> tuple[CodeAct, _ScriptedProxy]:
    proxy = _ScriptedProxy(replies)
    agent = CodeAct(
        llm_proxy=proxy,
        max_num_turns=4,
        tools=[_Ping()],
        bus=EventBus(listeners=[]),
        persistent_state=False,
        max_tool_calls=CAP,
        task_family="knapsack",
        **kwargs,
    )
    return agent, proxy


def _system(agent: CodeAct) -> list[str]:
    return [m["content"] for m in agent.message_history if m["role"] == "system"]


async def _observations(**kwargs) -> list[dict]:
    agent, _ = _agent(**kwargs)
    await agent.run("count the pings")
    return [
        json.loads(m["content"])
        for m in agent.message_history
        if m["role"] == "user" and "observation" in m["content"]
    ]


# --------------------------------------------------------------------------------------
# carryover_fewshot
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        dict(state_carryover_on_cap=True),
        dict(state_carryover_on_cap=True, announce_carryover=True),
        dict(state_carryover_on_cap=True, announce_carryover=True, resumption_metadata=True),
    ],
)
@pytest.mark.asyncio
async def test_unset_carryover_fewshot_leaves_existing_cells_unchanged(flags):
    default, _ = _agent(**flags)
    explicit, _ = _agent(**flags, carryover_fewshot=flags.get("announce_carryover", False))
    assert _system(default) == _system(explicit)


@pytest.mark.asyncio
async def test_announced_run_without_the_example():
    with_example, _ = _agent(state_carryover_on_cap=True, announce_carryover=True)
    without, _ = _agent(
        state_carryover_on_cap=True, announce_carryover=True, carryover_fewshot=False
    )
    assert L2_EXAMPLE in _system(with_example)
    assert L2_EXAMPLE not in _system(without)
    # the ONLY difference is the dropped example
    assert [s for s in _system(with_example) if s != L2_EXAMPLE] == _system(without)
    assert STATELESS_FEWSHOT_EXAMPLES in _system(without)


@pytest.mark.asyncio
async def test_example_without_the_announcement():
    silent, _ = _agent(state_carryover_on_cap=True)
    shown, _ = _agent(state_carryover_on_cap=True, carryover_fewshot=True)
    assert [s for s in _system(shown) if s != L2_EXAMPLE] == _system(silent)
    assert L2_EXAMPLE in _system(shown) and L3_EXAMPLE not in _system(shown)

    # ...and the banner stays the silent one: nothing is announced
    obs = await _observations(state_carryover_on_cap=True, carryover_fewshot=True)
    assert all(o["runtime_state"]["active_globals"] == [] for o in obs)


# --------------------------------------------------------------------------------------
# null_last_action
# --------------------------------------------------------------------------------------

L3_FLAGS = dict(state_carryover_on_cap=True, announce_carryover=True, resumption_metadata=True)


@pytest.mark.asyncio
async def test_null_last_action_keeps_the_prompt_and_key_but_never_fills_it():
    real = await _observations(**L3_FLAGS)
    nulled = await _observations(**L3_FLAGS, null_last_action=True)

    # the real L3 cell does report where the truncated turn had got to
    assert real[0]["runtime_state"]["last_action"] is not None
    for o in nulled:
        assert "last_action" in o["runtime_state"]
        assert o["runtime_state"]["last_action"] is None
    # the banner is otherwise untouched
    assert [o["runtime_state"]["active_globals"] for o in real] == [
        o["runtime_state"]["active_globals"] for o in nulled
    ]

    real_agent, _ = _agent(**L3_FLAGS)
    nulled_agent, _ = _agent(**L3_FLAGS, null_last_action=True)
    assert _system(real_agent) == _system(nulled_agent)


# --------------------------------------------------------------------------------------
# placeholder_tool_examples
# --------------------------------------------------------------------------------------


def _knapsack_tools():
    env = KnapsackEnv.__new__(KnapsackEnv)  # tools only hold a reference
    return env, [InspectTool(env)]


def test_default_inspect_doc_is_unchanged():
    _, (tool,) = _knapsack_tools()
    assert (
        """  inspect("item_0263322e3f1c") -> '{"class":"A","value":13,"weight":12}'\n"""
        in tool.doc
    )


def test_placeholder_swaps_only_the_example_line():
    _, (before,) = _knapsack_tools()
    _, tools = _knapsack_tools()
    (after,) = use_placeholder_tool_examples("knapsack", tools)
    assert INSPECT_DOC_EXAMPLE not in after.doc
    assert INSPECT_DOC_EXAMPLE_PLACEHOLDER in after.doc
    assert after.doc == before.doc.replace(
        INSPECT_DOC_EXAMPLE, INSPECT_DOC_EXAMPLE_PLACEHOLDER
    )
    # a fresh tool is untouched: the swap is per instance, not on the class
    _, (fresh,) = _knapsack_tools()
    assert INSPECT_DOC_EXAMPLE in fresh.doc


def test_placeholder_is_knapsack_only():
    with pytest.raises(ValueError):
        use_placeholder_tool_examples("navigation", [])


# --------------------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------------------


def _config(**overrides):
    from codeact_runtime.benchmark.config import AgentConfig

    return AgentConfig.model_validate(
        {
            "name": "x",
            "llm": {"model": "openai/persistent"},
            "max_tool_calls": CAP,
            "persistent_state": False,
            **overrides,
        }
    )


def test_config_rejects_meaningless_combinations():
    with pytest.raises(ValueError):
        _config(carryover_fewshot=True)  # no carryover to show an example of
    with pytest.raises(ValueError):
        _config(state_carryover_on_cap=True, announce_carryover=True, null_last_action=True)
    # the ablation cells themselves are accepted
    _config(state_carryover_on_cap=True, carryover_fewshot=True)
    _config(state_carryover_on_cap=True, announce_carryover=True, carryover_fewshot=False)
    _config(**L3_FLAGS, null_last_action=True)
    _config(placeholder_tool_examples=True)
