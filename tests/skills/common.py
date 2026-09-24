"""Builders for synthetic, recorded traces used by the skills tests."""

from __future__ import annotations

from pathlib import Path

from crystallizer.redaction import Redactor
from crystallizer.schemas import (
    Action,
    ArgValue,
    Scalar,
    Situation,
    StepResult,
    TraceStep,
    TraceVerdict,
)
from crystallizer.traces import TraceRecorder

NAMES = ["parser", "lexer", "emitter", "checker", "loader", "writer", "reader", "mapper"]


def make_step(
    run_id: str,
    task_id: str,
    index: int,
    kind: str,
    action: Action,
    params: dict[str, Scalar],
    *,
    attempt: int = 1,
    task_kind: str = "add_module",
    route: str = "small",
    verified: bool = True,
    ok: bool = True,
    redacted: bool = False,
) -> TraceStep:
    """One trace step."""
    return TraceStep(
        id=f"{run_id}:{task_id}:{attempt}:{index}",
        run_id=run_id,
        task_id=task_id,
        attempt=attempt,
        index=index,
        ts="2026-01-01T00:00:00.000000Z",
        situation=Situation(task_kind=task_kind, step_kind=kind, step_index=index, params=params),
        action=action,
        result=StepResult(ok=ok, output_hash="h"),
        route=route,
        verified=verified,
        redacted=redacted,
        tokens_in=100,
        tokens_out=50,
        cost=0.001,
        model_calls=1,
    )


def act(tool: str, **args: ArgValue) -> Action:
    """Shorthand for an action."""
    return Action(tool=tool, args=args)


def module_task(
    run_id: str, name: str, body: str, *, commit_message: str | None = None
) -> list[TraceStep]:
    """scaffold (mechanical) → body (model-authored) → add, commit (mechanical)."""
    task_id = f"add-{name}"
    params: dict[str, Scalar] = {"name": name}
    message = commit_message if commit_message is not None else f"add {name}"
    return [
        make_step(
            run_id,
            task_id,
            0,
            "scaffold",
            act(
                "file_write",
                path=f"src/{name}.py",
                content=f"def {name}():\n    raise NotImplementedError\n",
            ),
            params,
        ),
        make_step(
            run_id,
            task_id,
            1,
            "implement",
            act("file_write", path=f"src/{name}.py", content=body),
            params,
        ),
        make_step(
            run_id, task_id, 2, "stage", act("shell", argv=["git", "add", f"src/{name}.py"]), params
        ),
        make_step(
            run_id,
            task_id,
            3,
            "commit",
            act("shell", argv=["git", "commit", "-m", message]),
            params,
        ),
    ]


def run_steps(run_id: str, names: list[str], seed: int = 0) -> list[TraceStep]:
    """One run's steps: one module task per name, each with a unique body."""
    steps: list[TraceStep] = []
    for offset, name in enumerate(names):
        body = f"def {name}():\n    return {1000 + seed * 100 + offset}\n"
        steps.extend(module_task(run_id, name, body))
    return steps


def record(directory: Path, steps: list[TraceStep], *, passed: bool = True) -> None:
    """Write steps plus a verdict per task attempt to trace files (verified via verdicts)."""
    by_run: dict[str, list[TraceStep]] = {}
    for step in steps:
        by_run.setdefault(step.run_id, []).append(step)
    for run_id, run in by_run.items():
        recorder = TraceRecorder(directory, run_id, Redactor())
        attempts: dict[tuple[str, int], list[str]] = {}
        for step in run:
            recorder.record_step(step.model_copy(update={"verified": False}))
            attempts.setdefault((step.task_id, step.attempt), []).append(step.id)
        for (task_id, attempt), ids in attempts.items():
            recorder.record_verdict(
                TraceVerdict(
                    run_id=run_id,
                    task_id=task_id,
                    attempt=attempt,
                    passed=passed,
                    ts="t",
                    step_ids=ids,
                )
            )
