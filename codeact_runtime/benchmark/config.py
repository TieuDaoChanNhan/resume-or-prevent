from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from codeact_runtime.config import (
    LLMConfig,
)
from codeact_runtime.families.knapsack import KnapsackEnv
from codeact_runtime.families.navigation import CEGISNavEnv
from codeact_runtime.families.rule_diagnosis import RuleDiagnosisEnv

FAMILY_ENVS = {
    "knapsack": KnapsackEnv,
    "navigation": CEGISNavEnv,
    "rule_diagnosis": RuleDiagnosisEnv,
}


class AgentConfig(BaseModel):
    name: str
    agent_type: Literal["codeact", "react"] = "codeact"
    llm: LLMConfig
    max_turns: int = Field(12, ge=1)
    timeout_s: float = Field(300.0, ge=1.0)
    persistent_state: bool = True
    max_tool_calls: int | None = None
    # Third runtime condition (cap-boundary carryover): reset every turn EXCEPT after one
    # the tool-call cap truncated, whose globals survive into the next turn only. Requires
    # persistent_state: false -- a persistent runtime keeps state at every boundary anyway.
    state_carryover_on_cap: bool = False
    # Announced variant: the runtime_state banner reports the bindings a carried checkpoint
    # leaves live. The only variant that changes what the policy READS.
    announce_carryover: bool = False
    # Fourth runtime condition (resumption metadata, L3): the same announced banner as
    # above, plus the name and arguments of the last tool call that completed before the
    # cap truncated the block. L2 tells the policy what survived; this also tells it where
    # it had gotten to. Requires announce_carryover -- it extends that banner with one
    # more field, it is not a standalone variant.
    resumption_metadata: bool = False
    # Ablations of what the announced variants bundle. carryover_fewshot left unset
    # follows announce_carryover (the worked carryover example rides with the banner);
    # set, it shows the example without the banner or withholds it from an announced run.
    carryover_fewshot: bool | None = None
    # L3 prompt and key, but last_action is always null: is it its content that matters?
    null_last_action: bool = False
    # Knapsack only: replace the concrete numbers in the inspect() doc example with type
    # placeholders, to test whether fabricated item values are copied from that example.
    placeholder_tool_examples: bool = False
    # Phase 2 teacher-trace strategy: "prevention" or "recovery" (see agent.py's
    # STRATEGY_PROMPTS), or None (default) for no strategy message at all.
    strategy_prompt: Literal["prevention", "recovery"] | None = None
    # Prevention only: the per-block tool-call budget the teacher is told to stay under.
    # Must be below max_tool_calls, or the instruction would be a no-op.
    strategy_block_calls: int | None = None

    @model_validator(mode="after")
    def validate_carryover(self) -> "AgentConfig":
        if self.carryover_fewshot and not self.state_carryover_on_cap:
            raise ValueError(
                "carryover_fewshot requires state_carryover_on_cap: the example shows "
                "a grant this runtime would never make."
            )
        if self.null_last_action and not self.resumption_metadata:
            raise ValueError(
                "null_last_action requires resumption_metadata: there is no "
                "last_action field to null out."
            )
        if self.announce_carryover and not self.state_carryover_on_cap:
            raise ValueError(
                "announce_carryover requires state_carryover_on_cap: there would be "
                "nothing surviving to announce."
            )
        if self.resumption_metadata and not self.announce_carryover:
            raise ValueError(
                "resumption_metadata requires announce_carryover: it extends the "
                "announced banner with the last completed tool call, there would be "
                "no banner to extend."
            )
        if self.state_carryover_on_cap:
            if self.persistent_state:
                raise ValueError(
                    "state_carryover_on_cap requires persistent_state: false."
                )
            if self.max_tool_calls is None:
                raise ValueError(
                    "state_carryover_on_cap needs a per-turn cap: with max_tool_calls "
                    "unset no turn is ever truncated, so the condition is a no-op."
                )
            if self.agent_type != "codeact":
                raise ValueError("state_carryover_on_cap is implemented for codeact.")
        if self.strategy_prompt == "prevention" and self.strategy_block_calls is None:
            raise ValueError(
                "strategy_prompt='prevention' requires strategy_block_calls."
            )
        if self.strategy_block_calls is not None:
            if self.strategy_prompt != "prevention":
                raise ValueError(
                    "strategy_block_calls only applies to strategy_prompt='prevention'."
                )
            if (
                self.max_tool_calls is not None
                and self.strategy_block_calls >= self.max_tool_calls
            ):
                raise ValueError(
                    "strategy_block_calls must be below max_tool_calls, or the "
                    "prevention instruction would be a no-op."
                )
        return self


class BenchmarkConfig(BaseModel):
    output_dir: Path
    run_name: str = "codeact"
    seed_start: int = 0
    max_concurrent_runs: int = Field(4, ge=1)
    cache_db_path: Path
    agents: list[AgentConfig]
    task_families: list[str] | None = None

    @field_validator("task_families")
    @classmethod
    def validate_task_families(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("task_families must contain at least one family.")
        invalid = sorted(set(value) - set(FAMILY_ENVS.keys()))
        if invalid:
            raise ValueError(
                "task_families must be one of: "
                + ", ".join(sorted(FAMILY_ENVS.keys()))
                + f". Invalid: {', '.join(invalid)}"
            )
        return value
