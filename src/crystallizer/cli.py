"""Command-line interface (argparse). A thin layer over :mod:`crystallizer.api`.

Exit codes: 0 success, 1 general error, 2 usage error, 3 a task failed, 4 human approval
unavailable or denied, 5 workspace locked, 6 checkpoint unrecoverable, 7 budget exhausted.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from crystallizer import __version__
from crystallizer.api import Harness
from crystallizer.bench import run_bench
from crystallizer.clock import SystemClock
from crystallizer.errors import CrystallizerError, ExitCode
from crystallizer.hashing import to_jsonable
from crystallizer.logging_setup import setup_logging
from crystallizer.redaction import default_redactor
from crystallizer.scenario import find_scenario, load_scenario
from crystallizer.schemas import (
    PreviewStep,
    RunSummary,
    SkillManifest,
    check_schemas,
    export_schemas,
)

Handler = Callable[[argparse.Namespace, TextIO], int]


def _emit(args: argparse.Namespace, out: TextIO, payload: Any, text: str) -> None:
    if getattr(args, "json", False):
        out.write(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    else:
        out.write(text.rstrip("\n") + "\n")


def open_harness(args: argparse.Namespace) -> Harness:
    """Build a harness from the global flags (wall clock, random seed: CLI only)."""
    return Harness.open(
        args.workspace,
        config_path=args.config,
        profile=args.profile,
        clock=SystemClock(),
        seed=secrets.randbits(32),
        dry_run=args.dry_run,
    )


def _preview_text(lines: list[PreviewStep]) -> str:
    if not lines:
        return "nothing to run"
    return "\n".join(
        f"{line.task_id} #{line.step_index} [{line.step_kind}] {line.route}: {line.detail}"
        for line in lines
    )


def _summary_text(summary: RunSummary) -> str:
    if summary.nothing_to_do:
        return "nothing to resume"
    lines = [f"run {summary.run_id}" + (" (resumed)" if summary.resumed else "")]
    lines.extend(f"  {t.state.value:8} {t.id}  attempts={t.attempts}" for t in summary.tasks)
    mix = ", ".join(f"{route}={count}" for route, count in summary.route_mix.items()) or "none"
    lines.append(f"  steps={summary.steps} routes: {mix}")
    lines.append(
        f"  tokens in/out={summary.usage.tokens_in}/{summary.usage.tokens_out} "
        f"cost={summary.usage.cost:.6f} model_calls={summary.usage.model_calls}"
    )
    return "\n".join(lines)


def cmd_init(args: argparse.Namespace, out: TextIO) -> int:
    """``init``: write the default config and create the state directory."""
    with open_harness(args) as harness:
        report = harness.init()
    verb = "would create" if report.dry_run else "created"
    body = "\n".join(f"  {item}" for item in report.created) or "  (nothing)"
    _emit(args, out, report, f"{verb}:\n{body}")
    return ExitCode.OK


def cmd_plan(args: argparse.Namespace, out: TextIO) -> int:
    """``plan GOAL``: ask the planner for a task DAG."""
    with open_harness(args) as harness:
        report = harness.plan(args.goal)
    if report.plan is None:
        _emit(args, out, report, report.detail)
        return ExitCode.OK
    lines = [f"plan for: {report.plan.goal}"]
    for task in report.plan.tasks:
        deps = f" after {', '.join(task.depends_on)}" if task.depends_on else ""
        lines.append(f"  {task.id}: {task.title} [{len(task.steps)} steps]{deps}")
    lines.append(f"  planning cost={report.usage.cost:.6f}")
    _emit(args, out, report, "\n".join(lines))
    return ExitCode.OK


def _run_or_resume(args: argparse.Namespace, out: TextIO, resume: bool) -> int:
    with open_harness(args) as harness:
        if args.dry_run:
            lines = harness.preview()
            _emit(args, out, {"dry_run": True, "steps": lines}, _preview_text(lines))
            return ExitCode.OK
        summary = harness.resume() if resume else harness.run()
    _emit(args, out, summary, _summary_text(summary))
    return summary.exit_code


def cmd_run(args: argparse.Namespace, out: TextIO) -> int:
    """``run``: execute the plan's unfinished tasks."""
    return _run_or_resume(args, out, resume=False)


def cmd_resume(args: argparse.Namespace, out: TextIO) -> int:
    """``resume``: continue an interrupted run."""
    return _run_or_resume(args, out, resume=True)


def cmd_status(args: argparse.Namespace, out: TextIO) -> int:
    """``status``: plan, run cursor and task states (read-only)."""
    with open_harness(args) as harness:
        report = harness.status()
    if not report.has_plan:
        _emit(args, out, report, "no plan yet: run `crystallizer plan GOAL`")
        return ExitCode.OK
    lines = [
        f"goal: {report.goal}",
        f"run: {report.run_id or '-'} ({'active' if report.run_active else 'idle'}), "
        f"runs recorded: {report.runs}",
    ]
    lines.extend(f"  {t.state.value:8} {t.id}  attempts={t.attempts}" for t in report.tasks)
    if report.skills:
        lines.append("skills: " + ", ".join(f"{k}={v}" for k, v in report.skills.items()))
    _emit(args, out, report, "\n".join(lines))
    return ExitCode.OK


def cmd_tools(args: argparse.Namespace, out: TextIO) -> int:
    """``tools list``: tool descriptors with JSON Schema arguments."""
    with open_harness(args) as harness:
        specs = harness.tool_specs()
    text = "\n".join(
        f"  {spec.name:13} {'reversible' if spec.reversible else 'irreversible':12} "
        f"{spec.description}"
        for spec in specs
    )
    _emit(args, out, specs, text)
    return ExitCode.OK


def _skill_line(entry: SkillManifest) -> str:
    return (
        f"  {entry.key:24} {entry.status.value:9} shadow={entry.shadow_passes}/{entry.shadow_runs}"
        f" live={entry.live_passes}/{entry.live_runs}  {entry.guard_summary}"
    )


MUTATING_SKILL_COMMANDS = frozenset({"promote", "demote", "mine", "evaluate", "import"})


def cmd_skills(args: argparse.Namespace, out: TextIO) -> int:
    """``skills list|show|promote|demote|mine|evaluate|export|import``."""
    action = args.skills_command
    with open_harness(args) as harness:
        if args.dry_run and action in MUTATING_SKILL_COMMANDS:
            target = getattr(args, "ref", None) or getattr(args, "file", None) or ""
            detail = f"dry run: would {action} {target}".rstrip()
            _emit(args, out, {"dry_run": True, "detail": detail}, detail)
            return ExitCode.OK
        if action == "list":
            entries = harness.skills_list()
            text = "\n".join(_skill_line(entry) for entry in entries) or "no skills yet"
            _emit(args, out, entries, text)
        elif action in ("show", "export"):
            payload = (
                harness.skill_show(args.ref) if action == "show" else harness.skill_export(args.ref)
            )
            out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        elif action == "promote":
            entry = harness.skill_promote(args.ref, force=args.force, reason=args.reason)
            _emit(args, out, entry, f"promoted {entry.key}")
        elif action == "demote":
            entry = harness.skill_demote(args.ref, reason=args.reason or "manual demotion")
            _emit(args, out, entry, f"demoted {entry.key}")
        elif action == "mine":
            added = harness.skills_mine()
            text = "\n".join(_skill_line(entry) for entry in added) or "no new candidates"
            _emit(args, out, added, text)
        elif action == "evaluate":
            count = harness.skills_evaluate()
            _emit(args, out, {"new_observations": count}, f"{count} new shadow observation(s)")
        else:
            imported = harness.skill_import(Path(args.file))
            text = f"imported {imported.key} as candidate" if imported else "already known"
            _emit(args, out, imported, text)
    return ExitCode.OK


def cmd_report(args: argparse.Namespace, out: TextIO) -> int:
    """``report``: cost by route, skill counts, estimated saving vs an all-large baseline."""
    with open_harness(args) as harness:
        report = harness.report()
    lines = [f"runs={report.runs} steps={report.steps}"]
    for route, totals in report.by_route.items():
        lines.append(
            f"  {route:8} steps={totals.steps:5} calls={totals.model_calls:5} "
            f"tokens={totals.tokens_in}/{totals.tokens_out} cost={totals.cost:.6f}"
        )
    lines.append(f"  overhead cost={report.overhead.cost:.6f}")
    lines.append(f"total cost={report.total.cost:.6f} model_calls={report.total.model_calls}")
    lines.append(
        f"all-large baseline={report.baseline_cost:.6f} "
        f"estimated saving={report.estimated_saving:.6f} ({report.saving_pct:.1f}%)"
    )
    lines.append("skills: " + ", ".join(f"{k}={v}" for k, v in report.skills.items()))
    lines.append(f"note: {report.note}")
    _emit(args, out, report, "\n".join(lines))
    return ExitCode.OK


def scenario_dirs(workspace: Path) -> list[Path]:
    """Where ``bench --scenario NAME`` looks: workspace, current directory, source checkout."""
    source_checkout = Path(__file__).resolve().parents[2] / "scenarios"
    candidates = [workspace / "scenarios", Path.cwd() / "scenarios", source_checkout]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate.resolve() not in [u.resolve() for u in unique]:
            unique.append(candidate)
    return unique


def cmd_bench(args: argparse.Namespace, out: TextIO) -> int:
    """``bench --scenario NAME --runs N``: reproducible MockModel benchmark (JSON output)."""
    scenario = load_scenario(find_scenario(args.scenario, scenario_dirs(args.workspace)))
    if args.dry_run:
        detail = f"dry run: would run {scenario.name} {args.runs} time(s)"
        _emit(args, out, {"dry_run": True, "detail": detail}, detail)
        return ExitCode.OK
    if args.keep:
        root = Path(args.keep)
        root.mkdir(parents=True, exist_ok=True)
        report = run_bench(scenario, args.runs, root, profile=args.profile)
    else:
        with tempfile.TemporaryDirectory(prefix="crystallizer-bench-") as tmp:
            report = run_bench(scenario, args.runs, Path(tmp), profile=args.profile)
    out.write(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return ExitCode.OK


def cmd_schemas(args: argparse.Namespace, out: TextIO) -> int:
    """``schemas export|check``: regenerate or verify the committed JSON Schemas."""
    directory = Path(args.dir)
    if args.schemas_command == "export":
        written = [path.name for path in export_schemas(directory)]
        _emit(args, out, {"written": written}, f"wrote {len(written)} schemas to {directory}")
        return ExitCode.OK
    problems = check_schemas(directory)
    if problems:
        _emit(
            args,
            out,
            {"ok": False, "problems": problems},
            "schema drift detected:\n" + "\n".join(f"  {p}" for p in problems),
        )
        return ExitCode.GENERAL
    _emit(args, out, {"ok": True, "problems": []}, "schemas are up to date")
    return ExitCode.OK


def add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add the global flags; with ``suppress`` they only override when given explicitly."""

    def default(value: object) -> object:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument(
        "--config", type=Path, default=default(None), help="config file (crystallizer.toml)"
    )
    parser.add_argument("--profile", default=default(None), help="apply [profile.NAME] overrides")
    parser.add_argument(
        "--workspace", type=Path, default=default(Path(".")), help="workspace directory"
    )
    for flag, text in (
        ("--verbose", "log at INFO level"),
        ("--debug", "log at DEBUG level"),
        ("--dry-run", "no tools, no state, no models"),
        ("--json", "machine-readable output"),
    ):
        parser.add_argument(flag, action="store_true", default=default(False), help=text)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with global flags and all subcommands."""
    parser = argparse.ArgumentParser(prog="crystallizer", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"crystallizer {__version__}")
    add_global_flags(parser, suppress=False)
    # The same flags are accepted after the subcommand (e.g. `bench ... --profile e2e`).
    common = argparse.ArgumentParser(add_help=False)
    add_global_flags(common, suppress=True)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init", parents=[common], help="write default config and create state directory"
    )
    init.set_defaults(handler=cmd_init)

    plan = commands.add_parser("plan", parents=[common], help="plan a goal into a task DAG")
    plan.add_argument("goal", help="the goal to plan")
    plan.set_defaults(handler=cmd_plan)

    run = commands.add_parser("run", parents=[common], help="run the plan's unfinished tasks")
    run.set_defaults(handler=cmd_run)

    resume = commands.add_parser("resume", parents=[common], help="continue an interrupted run")
    resume.set_defaults(handler=cmd_resume)

    status = commands.add_parser("status", parents=[common], help="show plan and run status")
    status.set_defaults(handler=cmd_status)

    tools = commands.add_parser("tools", parents=[common], help="list tools")
    tools.add_argument("tools_command", choices=["list"])
    tools.set_defaults(handler=cmd_tools)

    bench = commands.add_parser(
        "bench", parents=[common], help="run a benchmark scenario with MockModel"
    )
    bench.add_argument("--scenario", required=True, help="scenario name or .toml path")
    bench.add_argument("--runs", type=int, default=5, help="number of runs (default 5)")
    bench.add_argument("--keep", help="keep workspaces and state in this directory")
    bench.set_defaults(handler=cmd_bench)

    report = commands.add_parser(
        "report", parents=[common], help="cost by route and estimated saving"
    )
    report.set_defaults(handler=cmd_report)

    skills = commands.add_parser("skills", parents=[common], help="inspect and manage skills")
    skill_commands = skills.add_subparsers(dest="skills_command", required=True)
    skill_commands.add_parser("list", parents=[common], help="list skills")
    for name, help_text in (("show", "show a skill"), ("export", "print skill JSON")):
        sub = skill_commands.add_parser(name, parents=[common], help=help_text)
        sub.add_argument("ref", help="skill id or id@vN")
    promote = skill_commands.add_parser("promote", parents=[common], help="promote a candidate")
    promote.add_argument("ref")
    promote.add_argument("--force", action="store_true", help="skip checks (needs --reason)")
    promote.add_argument("--reason", help="why (required with --force)")
    demote = skill_commands.add_parser("demote", parents=[common], help="demote an active skill")
    demote.add_argument("ref")
    demote.add_argument("--reason", help="why")
    skill_commands.add_parser("mine", parents=[common], help="mine recorded traces into candidates")
    skill_commands.add_parser(
        "evaluate", parents=[common], help="shadow-evaluate candidates on recorded traces"
    )
    importer = skill_commands.add_parser(
        "import", parents=[common], help="import skill JSON as a candidate"
    )
    importer.add_argument("file")
    skills.set_defaults(handler=cmd_skills)

    schemas = commands.add_parser("schemas", parents=[common], help="export or check JSON Schemas")
    schemas.add_argument("schemas_command", choices=["export", "check"])
    schemas.add_argument("--dir", default="schemas", help="schema directory (default: schemas)")
    schemas.set_defaults(handler=cmd_schemas)
    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run the CLI and return the process exit code."""
    stream = out or sys.stdout
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging(verbose=args.verbose, debug=args.debug)
    handler: Handler = args.handler
    try:
        return int(handler(args, stream))
    except CrystallizerError as exc:
        sys.stderr.write(f"error: {exc.message}\n")
        return int(exc.exit_code)
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return int(ExitCode.GENERAL)
    except Exception as exc:  # noqa: BLE001 - last-resort handler must redact and exit 1
        message = default_redactor().redact(f"{type(exc).__name__}: {exc}")
        sys.stderr.write(f"error: {message}\n")
        return int(ExitCode.GENERAL)
