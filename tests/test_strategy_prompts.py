"""Phase 2 teacher-trace strategy prompts (prevention / recovery).

strategy_prompt is orthogonal to every carryover flag: unset, it must leave every existing
cell's prompt byte-for-byte unchanged. Set, it adds exactly one system message, the last
one before the tool list, and that message's TEXT must not vary with the runtime (L0 vs
L3) -- only the live `active_globals`/`last_action` values the agent sees at runtime vary.
"""

import json

import pytest

from codeact_runtime.codeact.agent import (
    STRATEGY_PROMPT_PREVENTION,
    STRATEGY_PROMPT_RECOVERY,
    STRATEGY_PROMPTS,
    CodeAct,
)
from codeact_runtime.codeact.events import EventBus

from tests.test_cap_carryover import CAP, REPLIES, _Ping, _ScriptedProxy

L0_FLAGS = dict(persistent_state=False, max_tool_calls=CAP)
L3_FLAGS = dict(
    persistent_state=False,
    max_tool_calls=CAP,
    state_carryover_on_cap=True,
    announce_carryover=True,
    resumption_metadata=True,
    task_family="knapsack",
)


async def _agent(replies=REPLIES, **kwargs) -> CodeAct:
    proxy = _ScriptedProxy(replies)
    return CodeAct(
        llm_proxy=proxy, max_num_turns=4, tools=[_Ping()], bus=EventBus(listeners=[]), **kwargs
    )


def _system(agent: CodeAct) -> list[str]:
    return [m["content"] for m in agent.message_history if m["role"] == "system"]


@pytest.mark.asyncio
async def test_unset_leaves_l0_and_l3_cells_unchanged():
    for flags in (L0_FLAGS, L3_FLAGS):
        default = await _agent(**flags)
        explicit = await _agent(**flags, strategy_prompt=None)
        assert _system(default) == _system(explicit)


@pytest.mark.parametrize("runtime_flags", [L0_FLAGS, L3_FLAGS])
@pytest.mark.asyncio
async def test_recovery_is_added_as_the_last_system_message_before_tools(runtime_flags):
    without = await _agent(**runtime_flags)
    with_it = await _agent(**runtime_flags, strategy_prompt="recovery")
    sys_without, sys_with = _system(without), _system(with_it)
    assert sys_with[:-2] == sys_without[:-1]  # same prefix, tool_prompt still last
    assert sys_with[-2] == STRATEGY_PROMPT_RECOVERY
    assert sys_with[-1] == sys_without[-1]  # tool_prompt itself untouched


@pytest.mark.parametrize("runtime_flags", [L0_FLAGS, L3_FLAGS])
@pytest.mark.asyncio
async def test_prevention_is_added_with_the_block_budget_filled_in(runtime_flags):
    agent = await _agent(**runtime_flags, strategy_prompt="prevention", strategy_block_calls=20)
    text = _system(agent)[-2]
    assert "20 tool calls" in text
    assert "{max_calls_below_cap}" not in text
    assert "PREVENT TRUNCATION" in text


@pytest.mark.asyncio
async def test_recovery_prompt_text_is_identical_across_l0_and_l3():
    """The instructions are written to be correct under whatever active_globals/
    last_action actually say; the PROMPT TEXT itself must not hint which runtime it is."""
    l0 = _system(await _agent(**L0_FLAGS, strategy_prompt="recovery"))[-2]
    l3 = _system(await _agent(**L3_FLAGS, strategy_prompt="recovery"))[-2]
    assert l0 == l3 == STRATEGY_PROMPT_RECOVERY
    assert "L0" not in l0 and "L3" not in l0 and "persistent" not in l0.lower()


@pytest.mark.asyncio
async def test_prevention_requires_block_calls():
    with pytest.raises(ValueError):
        await _agent(**L0_FLAGS, strategy_prompt="prevention")


@pytest.mark.asyncio
async def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        await _agent(**L0_FLAGS, strategy_prompt="bogus")


@pytest.mark.asyncio
async def test_on_l0_recovery_sees_empty_active_globals_after_a_cut():
    """The behavioural claim behind the prompt design: on L0 there is genuinely nothing
    to recover, so active_globals stays empty even right after a cap cut -- recovery's
    own conditional ("if it is empty: rebuild from printed output") is what fires, not a
    hidden L3 assumption."""
    agent = await _agent(**L0_FLAGS, strategy_prompt="recovery")
    await agent.run("count the pings")
    obs = [
        json.loads(m["content"])
        for m in agent.message_history
        if m["role"] == "user" and "observation" in m["content"]
    ]
    assert any("Tool call limit" in (o["observation"]["error"] or "") for o in obs)
    assert all(o["runtime_state"]["active_globals"] == [] for o in obs)
    assert all("last_action" not in o["runtime_state"] for o in obs)


@pytest.mark.asyncio
async def test_on_l3_recovery_sees_the_real_banner_after_a_cut():
    agent = await _agent(**L3_FLAGS, strategy_prompt="recovery")
    await agent.run("count the pings")
    obs = [
        json.loads(m["content"])
        for m in agent.message_history
        if m["role"] == "user" and "observation" in m["content"]
    ]
    # the cut turn's OWN observation already reports what survives into the next one
    cut_turn = next(i for i, o in enumerate(obs) if "Tool call limit" in (o["observation"]["error"] or ""))
    assert obs[cut_turn]["runtime_state"]["active_globals"] != []


def test_config_validation():
    from codeact_runtime.benchmark.config import AgentConfig

    def build(**overrides):
        return AgentConfig.model_validate(
            {
                "name": "x",
                "llm": {"model": "openai/x"},
                "max_tool_calls": 25,
                "persistent_state": False,
                **overrides,
            }
        )

    with pytest.raises(ValueError):
        build(strategy_prompt="prevention")  # missing strategy_block_calls
    with pytest.raises(ValueError):
        build(strategy_block_calls=20)  # no strategy_prompt to apply it to
    with pytest.raises(ValueError):
        build(strategy_prompt="prevention", strategy_block_calls=25)  # not below cap
    with pytest.raises(ValueError):
        build(strategy_prompt="recovery", strategy_block_calls=20)  # recovery has no budget
    build(strategy_prompt="prevention", strategy_block_calls=20)
    build(strategy_prompt="recovery")
    build()
