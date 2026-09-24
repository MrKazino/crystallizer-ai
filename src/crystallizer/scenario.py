"""Benchmark scenarios: TOML files that generate a plan, a MockModel script and fixtures per run.

A scenario declares a task template instantiated ``tasks_per_run`` times per run. Parameters are
drawn per run, without replacement, from the ``[params]`` pools with a seeded generator, so slots
genuinely vary. ``{rand}`` is a seeded integer that is unique across the whole benchmark: it
stands in for model-authored content that no template can derive. Templates use the skill syntax
(``{name}``, ``{name|upper}``, ``{{`` for a literal brace) and are rendered by the same renderer.

Each step declares: kind, tool, args, ``mechanical`` (documentation of intent, checked by the
benchmark report), ``tokens_in``, ``tokens_out`` and ``small_correct`` (false makes the small
tier's samples disagree, forcing escalation to the large tier).
"""

from __future__ import annotations

import random
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from crystallizer.errors import ConfigError
from crystallizer.models import MockBehavior, MockScript, MockStep
from crystallizer.planner import validate_plan
from crystallizer.schemas import Action, ArgValue, Plan, Scalar, StepSpec, Task
from crystallizer.skills.executor import render_template

RAND_RANGE = (10_000, 99_999)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioStep(_Model):
    """One scripted step of the task template."""

    kind: str
    tool: str
    args: dict[str, ArgValue] = Field(default_factory=dict)
    mechanical: bool
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    small_correct: bool = True
    description: str = ""


class Fixture(_Model):
    """A file written into the workspace before the run (templated for per-task fixtures)."""

    path: str
    content: str


class Scenario(_Model):
    """A validated scenario file."""

    name: str
    description: str
    seed: int
    tasks_per_run: int = Field(ge=1)
    task_kind: str
    task_id: str
    title: str
    params: dict[str, list[str]]
    acceptance: list[list[str]] = Field(default_factory=list)
    fixture: list[Fixture] = Field(default_factory=list)
    task_fixture: list[Fixture] = Field(default_factory=list)
    step: list[ScenarioStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _pools_are_large_enough(self) -> Scenario:
        for name, pool in self.params.items():
            if len(set(pool)) < self.tasks_per_run:
                raise ValueError(f"param pool {name!r} needs {self.tasks_per_run} distinct values")
        return self


class RunSpec(BaseModel):
    """Everything one benchmark run needs."""

    model_config = ConfigDict(extra="forbid")

    plan: Plan
    script: MockScript
    fixtures: dict[str, str]
    mechanical_steps: int
    total_steps: int


def load_scenario(path: Path) -> Scenario:
    """Parse and validate a scenario TOML file."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return Scenario.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError(f"invalid scenario {path.name}: {exc}") from None


def find_scenario(name: str, search: list[Path]) -> Path:
    """Resolve a scenario by file path or by name inside the search directories."""
    direct = Path(name)
    if direct.suffix == ".toml" and direct.is_file():
        return direct
    for directory in search:
        candidate = directory / f"{name}.toml"
        if candidate.is_file():
            return candidate
    raise ConfigError(f"scenario {name!r} not found in: {', '.join(str(d) for d in search)}")


def _render(value: ArgValue, values: dict[str, Scalar]) -> ArgValue:
    if isinstance(value, str):
        return render_template(value, values)
    if isinstance(value, list):
        return [render_template(item, values) for item in value]
    return value


def instantiate(scenario: Scenario, rng: random.Random, used_rands: set[int]) -> RunSpec:
    """Draw one run's parameters and build its plan, script and fixtures."""
    chosen = {
        name: rng.sample(sorted(set(pool)), scenario.tasks_per_run)
        for name, pool in sorted(scenario.params.items())
    }
    tasks: list[Task] = []
    actions: dict[str, list[MockStep]] = {}
    fixtures = {fixture.path: fixture.content for fixture in scenario.fixture}
    for index in range(scenario.tasks_per_run):
        params: dict[str, Scalar] = {name: chosen[name][index] for name in sorted(chosen)}
        rand = rng.randint(*RAND_RANGE)
        while rand in used_rands:
            rand = rng.randint(*RAND_RANGE)
        used_rands.add(rand)
        values: dict[str, Scalar] = {**params, "index": index, "rand": rand}
        task_id = render_template(scenario.task_id, values)
        tasks.append(
            Task(
                id=task_id,
                title=render_template(scenario.title, values),
                kind=scenario.task_kind,
                params=params,
                steps=[
                    StepSpec(kind=step.kind, description=render_template(step.description, values))
                    for step in scenario.step
                ],
                acceptance_commands=[
                    [render_template(arg, values) for arg in command]
                    for command in scenario.acceptance
                ],
            )
        )
        actions[task_id] = [
            MockStep(
                action=Action(
                    tool=step.tool,
                    args={name: _render(arg, values) for name, arg in step.args.items()},
                ),
                tokens_in=step.tokens_in,
                tokens_out=step.tokens_out,
                small=MockBehavior.CORRECT if step.small_correct else MockBehavior.DISAGREE,
            )
            for step in scenario.step
        ]
        for fixture in scenario.task_fixture:
            fixtures[render_template(fixture.path, values)] = render_template(
                fixture.content, values
            )
    plan = validate_plan(Plan(goal=f"{scenario.name}: {scenario.description}", tasks=tasks))
    mechanical = sum(1 for step in scenario.step if step.mechanical) * scenario.tasks_per_run
    return RunSpec(
        plan=plan,
        script=MockScript(actions=actions),
        fixtures=fixtures,
        mechanical_steps=mechanical,
        total_steps=len(scenario.step) * scenario.tasks_per_run,
    )
