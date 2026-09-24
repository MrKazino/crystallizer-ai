# Notes

## Honest scope limits

1. Only mechanical steps can become skills: steps whose every argument is a constant or a value
   derivable from the situation. Steps needing model-authored content always go to a model.
2. The benchmark uses MockModel. It proves the routing, mining, shadow-testing, and promotion
   mechanics work. It does not prove cost savings with a real model.
3. The tool sandbox confines paths, executables, environment, and time. It is not OS-level
   isolation. Recommend running inside a container for untrusted work.
4. Cost defaults are illustrative units. Users must set real prices in config.

Additional limits introduced by v2 (see docs/SPEC.md section 1):

5. Guards are deliberately conservative: a parameter that varies but is not used by a skill
   restricts the skill to the observed values of that parameter.
6. Code run by tests or allowed executables runs with the user's privileges.
7. Plugin tiers and plugin model providers self-report their usage and cost.

## Decisions

1. **Spec v2.0** (`docs/SPEC.md`) supersedes v1.0; every amendment is tagged `[v2]` with a reason and
   listed in its Appendix A.
2. **Step model**: tasks carry a planned step skeleton; the situation is known before routing
   (`task_kind`, `step_kind`, `step_index`, `params`). Tasks without steps run in open mode.
3. **Slots bind to situation fields** (`{type, source}`), not bare names.
4. **Journal key includes the attempt number**; write-ahead `started`/`completed` with in-doubt
   handling: idempotent actions are redone, anything else (including `git commit`) goes to the
   human.
5. **Traces stay append-only**: `verified` is derived from verdict records at load time.
6. **Redaction precision**: left boundaries on token patterns, component-based sensitive keys,
   fixpoint iteration; `.env` values shorter than 4 characters are not treated as secrets.
7. **Tools cannot read `state_dir`** either (least privilege), and `.git` is write-protected.
8. **`python` resolves to the running interpreter** so tests and acceptance commands use the same
   environment regardless of `PATH`.
9. **Confidence of a single sample is capped at 0.5**, so allow-listed irreversible actions from a
   single-sample tier still escalate.
10. **Acceptance commands run through the sandbox and policy**, because they are model-authored.
11. **Acceptance escalation stops at the large tier**; the human tier approves or supplies actions.
12. **`run` refuses while a run is unfinished**; `resume` continues it with the same run id.
13. **Bench**: fresh workspace per run, persistent state directory, plan loaded from the scenario.
14. **Config and tool-argument models live next to their code** (`config.py`, `tools.py`); all
    persisted and interchange models live in `schemas.py`.
15. **Build backend**: `setuptools` pinned as a build-time requirement only.
16. **Mining templates** only use situation values of at least 3 characters on non-alphanumeric
    boundaries, and must round-trip exactly.
17. **Budget checks happen before each model call**; the call that crosses a limit completes.
18. **Promotion is evaluated at the end of each run**; demotion is immediate.
19. **Resume replays executed steps from the trace** (no model call) and the journal (no tool
    call); only redacted steps are routed again.
20. **In-doubt actions are redone only when idempotent** (`policy.is_idempotent`); a crashed
    `git commit` or any irreversible action asks the human.
21. **Usage that executes nothing is traced** as `overhead` records (failed routing, `done`,
    budget or human interruptions), so reports and resumed budgets stay exact.
22. **Run ids are unique per state directory**: the runner regenerates an id whose trace file
    already exists (deterministic, since the clock and RNG are injected).
23. **Scenario format**: a task template instantiated per run with parameters drawn without
    replacement from seeded pools; `{rand}` (unique across the benchmark) stands in for
    model-authored content; templates reuse the skill template syntax and renderer.
24. **Benchmark workspaces** are prepared by internal code (fixtures committed to a fresh git
    repository with a local identity and signing disabled); the agent never gets that access.
25. **`Harness.adopt_plan`** accepts plans from other planners or workflow engines; the benchmark
    uses it so planning cost does not blur the per-run comparison.
26. **Global CLI flags are accepted before or after the subcommand**, matching the spec's
    `bench --scenario NAME --runs N --profile e2e` form.
27. **AnthropicClient imports the SDK with `importlib`** at first use, so neither tests nor the
    default install ever import it, and mypy strict does not need its stubs.
28. **Tool options are allow-listed, not deny-listed**: `pytest`, `ruff` and `mypy` stay reversible
    only with known-safe options and workspace-relative paths (fail closed on anything else);
    `PATH` entries that are relative or inside the workspace are removed for subprocesses.

## Deviations

1. **LICENSE holder** is "Koosha KZ" instead of "Logic-Love" (v1 section 3). Reason: the repository
   owner chose this holder when asked. Impact: none on behavior; the proprietary wording is
   otherwise identical.
2. **add-modules has 5 steps per module (4 mechanical)** instead of 4 (3 mechanical). Reason: `git`
   cannot commit untracked files from a single argv without a shell, so staging is its own step.
   Impact: 20 steps per run with 16 mechanical (80%) instead of 16 with 12 (75%); the e2e
   threshold (run 5 ≤ 50% of run 1) is unchanged.

## Known risks

- The benchmark and the scenario acceptance commands need `git` and `pytest` on the machine
  (`make install` provides pytest).

- Real-model behavior (formatting drift, partial JSON) is only exercised through MockModel's
  `garbage` and `error` modes; a real provider may need prompt tuning.
- Path checks are resolve-then-open; a concurrent process could race a symlink swap between the
  two (mitigated by `O_NOFOLLOW` on the final component, not eliminated).
- Guard conservativeness can keep useful skills from firing when tasks carry extra varying params.
- SQLite WAL on network filesystems is not supported by SQLite; keep `state_dir` local.
