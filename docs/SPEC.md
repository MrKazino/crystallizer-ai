# crystallizer-ai: Specification v2.0

An agent harness that gets cheaper the longer it runs. It keeps context small, survives crashes,
and compiles repeated mechanical agent behavior into declarative, tested skills, routing every
step down the cheapest safe path. Any agent system (a model, an external agent framework, a rules
engine, a human) can be plugged in as a rung of that path and inherits the same safety envelope.

Names: repo and distribution `crystallizer-ai`, import name `crystallizer`, CLI command
`crystallizer`. Python 3.11 or newer.

**How to read this document.** v2.0 keeps every v1.0 requirement. Text tagged **[v2]** is an
addition or a precise amendment; each amendment states its reason. Appendix A lists every change
against v1.0 so the delta is auditable. Where this document and v1.0 differ, this document wins.

---

## 1. Honest scope limits (repeated verbatim in NOTES.md and README)

1. Only mechanical steps can become skills: steps whose every argument is a constant or a value
   derivable from the situation. Steps needing model-authored content always go to a model.
2. The benchmark uses MockModel. It proves the routing, mining, shadow-testing, and promotion
   mechanics work. It does not prove cost savings with a real model.
3. The tool sandbox confines paths, executables, environment, and time. It is not OS-level
   isolation. Recommend running inside a container for untrusted work.
4. Cost defaults are illustrative units. Users must set real prices in config.

**[v2] Additional limits** (also in NOTES.md):

5. Guards are deliberately conservative. A situation parameter that varies across occurrences but
   is not used by the skill restricts the guard to the observed values, so the skill will not fire
   for new values of that parameter. This is fail-closed by design.
6. Code executed by `run_tests`, `pytest` or other allowed executables runs with the user's
   privileges. The sandbox governs what the *harness* does, not what that code does.
7. Plugin tiers and plugin model providers self-report their token usage and cost.

## 2. Operating rules for the implementer

1. Read this whole file before writing any code. Work one phase at a time, in order.
2. After each phase: run `make check`, fix every failure, commit with a clear message, then print
   a summary: what exists, what is next, known gaps. (Whether to pause for review between phases
   is the repository owner's call; the summary and commit are mandatory.)
3. Complete code only. No stubs, TODOs, placeholders, `pass` bodies, or "left as exercise".
4. Edit files in place. Never rewrite a whole file to change a few lines.
5. Tests are deterministic and need no network, no API key, and no wall-clock dependence. All
   model access goes through the `ModelClient` protocol. Tests use `MockModel`. Inject clocks and
   random seeds; never call `time.time()` or `random` directly in logic.
6. No secrets in code, logs, traces, memory, checkpoints, prompts or error messages. Redact by
   default.
7. Dependencies: stdlib, pydantic, pytest, pytest-cov, hypothesis, ruff, mypy. The `anthropic`
   package is an optional extra named `anthropic` and is never imported by tests or by default.
   Pin exact versions in pyproject.toml (latest stable at implementation time, verified to
   install). No floating ranges. **[v2]** The build backend (`setuptools`, pinned) is a build-time
   requirement only.
8. Type hints and docstrings on every public function, class, and module. mypy strict.
9. If the spec is ambiguous, choose the simplest safe option, record it in NOTES.md under
   "Decisions", and continue.
10. If any requirement cannot be met, do not skip it silently. Record it in NOTES.md under
    "Deviations" with the reason and the impact.
11. Never claim performance or savings that a test or benchmark does not measure.
12. Never execute, evaluate, or import generated code. Skills are data, not code. **[v2]** Plugins
    are installed packages the user explicitly names in config; they are not generated code.

## 3. Repository layout

    pyproject.toml            metadata, pinned deps, ruff/mypy/pytest config, extras
    Makefile                  install, lint, typecheck, test, schemas, check
    README.md  NOTES.md  CONTRIBUTING.md  LICENSE  .gitignore
    docs/SPEC.md              this document                                   [v2]
    docs/ARCHITECTURE.md      layers, data flow, state machines, trust model  [v2]
    docs/EXTENDING.md         plugging in agents, providers, tools, observers [v2]
    docs/ROADMAP.md           phase checklists and status                     [v2]
    .github/workflows/ci.yml  runs make check on Python 3.11 and 3.12
    schemas/                  generated JSON Schemas (committed)
    scenarios/                bench scenario TOML files (3)
    src/crystallizer/
      __init__.py  __main__.py  py.typed
      errors.py         exception hierarchy mapped 1:1 to exit codes         [v2]
      clock.py          injected Clock, seeded RNG, run ids                  [v2]
      hashing.py        canonical JSON and SHA-256                           [v2]
      faults.py         deterministic fault injection for crash tests        [v2]
      schemas.py        all persistent/interchange pydantic models, schema export and drift check
      config.py         TOML config, profiles, validation, defaults
      logging_setup.py  structured JSON logging with redaction filter
      redaction.py      redaction patterns and helpers
      models.py         ModelClient protocol, MockModel, AnthropicClient, cost accounting
      budget.py         per-run budget governor                              [v2]
      db.py             single SQLite connection manager and migrations      [v2]
      lock.py           exclusive per-workspace run lock
      memory.py         SQLite project memory
      context.py        token-budgeted context builder
      planner.py        goal to task DAG with acceptance checks
      journal.py        write-ahead idempotency journal for tool actions
      checkpoint.py     atomic save and restore of project state
      traces.py         append-only JSONL trace recorder and normalizer
      tools.py          sandboxed tools and the tool registry
      policy.py         action classification and irreversible-action policy
      events.py         event bus for observers                              [v2]
      extensions.py     registries and plugin loading                        [v2]
      tiers.py          Proposer protocol and built-in ladder tiers          [v2]
      approval.py       human approval (TTY, scripted, deny)                 [v2]
      skills/__init__.py
      skills/miner.py       find repeated mechanical sub-trajectories
      skills/compiler.py    pattern to declarative skill (JSON) with guard
      skills/executor.py    interpret a skill (no eval, no code generation)
      skills/shadow.py      shadow-test candidates against verified outcomes
      skills/registry.py    versioned store, manifest, promote and demote
      router.py         cost ladder and escalation logic
      runner.py         orchestrates plan, context, route, act, trace, checkpoint
      scenario.py       bench scenario loader and generator                  [v2]
      bench.py          reproducible benchmark harness
      api.py            Harness facade: the stable programmatic API          [v2]
      cli.py            argparse CLI (thin layer over api.py)
    tests/              mirrors src layout, plus tests/e2e

LICENSE contains exactly this text and nothing else **[v2: holder set by the repository owner]**:

    Copyright (c) 2026 Koosha KZ. All rights reserved.

    This source code is made publicly visible for viewing only.
    No permission is granted to use, copy, modify, merge, publish,
    distribute, sublicense, or sell any part of this software,
    in source or binary form, without prior written permission
    from the copyright holder.

Do not use MIT or any open-source license anywhere. In pyproject.toml use the classifier
`License :: Other/Proprietary License`. README states the license in one sentence.

.gitignore covers: `.crystallizer/`, `__pycache__/`, `.venv/`, `.coverage`, `htmlcov/`,
`.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `dist/`, `*.egg-info/`, `.env`
(**[v2]** plus `build/` and `.hypothesis/`).

## 4. Configuration

TOML file (default `crystallizer.toml` in the workspace, or `--config`). Validated by pydantic.
Unknown keys are errors, **including inside profiles that are not selected [v2]**. A missing
default file means defaults; **[v2]** a missing explicit `--config` file is a usage error.
`--profile NAME` applies `[profile.NAME]` overrides on top of the base values.

v1 defaults (unchanged):

    [workspace]
    state_dir = ".crystallizer"   # relative to the workspace, or absolute

    [model]
    provider = "mock"            # "mock", "anthropic", or a plugin provider name   [v2]
    small = "small"              # model identifiers, set by the user for real providers
    large = "large"
    max_tokens = 1024

    [cost]                       # per million tokens, illustrative units
    small_in = 0.25
    small_out = 1.25
    large_in = 3.0
    large_out = 15.0

    [context]
    budget_tokens = 6000
    chars_per_token = 4          # documented estimator: ceil(len(text) / chars_per_token)

    [router]
    samples = 3
    agreement = 0.67
    retries = 1
    irreversible_confidence = 0.9

    [skills]
    min_repeats = 3
    min_shadow_runs = 20
    promote_pass_rate = 0.98
    demote_window = 20
    demote_pass_rate = 0.95

    [tools]
    allowed_executables = ["python", "pytest", "ruff", "mypy", "git"]
    timeout_seconds = 60
    env_allowlist = ["PATH", "HOME", "LANG", "PYTHONPATH"]
    max_file_bytes = 1000000

    [policy]
    allow_irreversible = []      # explicit allow-list of action names, empty by default

    [profile.e2e]
    skills.min_repeats = 2
    skills.min_shadow_runs = 5

**[v2] Additional keys** (all optional, shown with defaults):

    [model]
    planner_tier = "large"       # tier used by the planner: "small" or "large"

    [context]
    memory_items = 5             # memory entries considered per context
    trace_items = 5              # recent trace lines considered per context

    [router]
    large_samples = 1            # samples drawn on the large tier
    ladder = ["skill", "small", "large", "human"]   # plugin tiers may be inserted by name

    [skills]
    auto_mine = true             # mine at the end of every run
    auto_promote = true          # evaluate promotion at the end of every run
    min_template_length = 3      # shortest situation value that may become a template slot

    [tools]
    max_output_bytes = 65536     # tool output truncation
    protected_paths = [".git"]   # writes and deletes here are rejected

    [policy]
    reversible_tools = []        # plugin tools trusted as reversible (they must also declare it)

    [budget]                     # hard per-run limits; 0 disables a limit
    max_cost_per_run = 0.0
    max_tokens_per_run = 0
    max_model_calls_per_run = 0

    [runner]
    max_open_steps = 12          # step cap for tasks planned without explicit steps

    [memory]
    half_life_days = 30.0        # recency decay for memory scoring

    [redaction]
    min_env_value_length = 4     # shorter .env values are not treated as secrets

    [plugins]
    enabled = []                 # entry-point plugins to load; nothing else is imported

The e2e profile is the only place thresholds are lowered. Tests never edit thresholds mid-run to
force an outcome. Ladder validation: tiers are unique; `skill`, when present, is first; `human`,
when present, is last; at least one non-human tier exists.

## 5. Data models (pydantic, in schemas.py; exported JSON Schemas in /schemas)

All models forbid unknown fields. `Scalar = str | int | float | bool`;
`ArgValue = str | int | bool | list[str]`.

- **Action**: tool (str), args (dict[str, ArgValue]). Frozen.
- **Situation**: task_kind, step_kind, **step_index [v2]**, params (dict of scalars). Field paths are
  dotted: `task_kind`, `step_kind`, `step_index`, `params.<name>`. Param names are identifiers.
- **StepSpec [v2]**: kind, description. The planned skeleton of a task.
- **Task**: id, title, kind, params, depends_on, acceptance_commands (argv lists), **steps [v2]**,
  state (pending|running|done|failed|blocked), **attempts [v2]**.
- **Plan**: schema_version, goal, tasks.
- **TraceStep**: id, run_id, task_id, **attempt [v2]**, **index [v2]**, ts, situation, action, result
  (ok, output_hash, checks_passed, **exit_code, summary [v2]**), route (tier name: skill|small|
  large|human or a plugin tier), escalation_reason, **escalation_path [v2]**, **skill_key [v2]**,
  tokens_in, tokens_out, cost, **model_calls [v2]**, verified, **redacted [v2]**, **replayed [v2]**.
- **TraceVerdict [v2]**: run_id, task_id, attempt, passed, ts, step_ids. Traces are append-only, so
  `verified` is derived from verdict records when traces are loaded.
- **TraceOverhead [v2]**: run_id, task_id, attempt, index, ts, reason, usage: model usage at a step
  position that executed nothing (failed routing, `done`, or an interruption by the budget or a
  human), so cost accounting stays exact.
- **Skill** (JSON on disk): schema_version, id, version, **task_kind [v2]**, slots, guard, steps.
  - **[v2]** A slot is `{type, source}`: its type and the situation field it binds to
    (e.g. `{"type": "identifier", "source": "params.name"}`). v1 said "name to type"; the source
    binding makes slot values unambiguous.
  - Slot types: `identifier`, `path`, `string`, `int`.
  - Guard: `{"all": [conditions]}`; each condition is field, op, value. Ops: `eq`, `in`, `range`
    (numeric min and max, inclusive), `has_type`, `exists`. No regex, no eval.
  - Step: tool, **step_kind [v2]**, args, where string args may contain `{slot}` or
    `{slot|filter}`; `{{` and `}}` are literal braces **[v2]**. Filters: `lower`, `upper`, `snake`.
- **SkillManifest**: id, version, **content_hash [v2]**, guard_summary, source_file, provenance
  (trace ids), **derived_runs [v2]**, **origin (mined|imported) [v2]**, pass_rate, shadow_runs,
  **shadow_passes, unsafe_diffs [v2]**, live_runs, **live_passes [v2]**, status
  (candidate|active|demoted), created_at, history (events with evidence).
- **RegistryManifest [v2]**: schema_version, registry_version, skills (list of SkillManifest).
- **Checkpoint**: schema_version, **seq [v2]**, **created_at [v2]**, plan, memory_snapshot_id,
  registry_version, cursor, hash. **Cursor [v2]**: run_id, active, task_id, attempt, floor,
  failures_at_floor.
- **PolicyDecision**: action_name, classification (reversible|irreversible), reason,
  **requires_human [v2]**.
- **[v2]** MemoryEntry, JournalEntry, ToolSpec, ToolResult, Usage, StepRequest, Proposal,
  TierResult, Event, Message, Completion.

Exported schemas: action, checkpoint, config, event, memory_entry, plan, policy_decision,
registry_manifest, situation, skill, skill_manifest, tool_spec, trace_overhead, trace_step,
trace_verdict.
`python -m crystallizer schemas check` fails if committed schemas differ from generated ones.

## 6. Module behavior

### 6.1 redaction.py and logging_setup.py
- Redact with `[REDACTED]`: `sk-[A-Za-z0-9_-]{16,}`, `gh[pousr]_[A-Za-z0-9]{20,}`,
  `AKIA[0-9A-Z]{16}`, `Bearer\s+\S+`, values of sensitive keys in `key=value`, `key: value`
  **and JSON [v2]** forms, and every value loaded from a `.env` file in the workspace.
- **[v2] Precision.** Token patterns require a left boundary (no preceding letter or digit), so
  `task-add-module-parser` is not an `sk-` key. A key is sensitive when one of its components (split
  on `_ . -` and camelCase) is `pwd`, ends with `password|passwd|secret|secrets|token|apikey`, or is
  `api` followed by `key`: `access_token`, `apiKey`, `GITHUB_TOKEN` are sensitive; `tokens_in`,
  `max_tokens` are not. `.env` values shorter than `min_env_value_length`, or contained in the
  marker, are skipped (they cannot be meaningfully secret and would corrupt unrelated text).
- Patterns are linear-time: possessive quantifiers, left-boundary look-behinds, and value scans
  that never overlap. **[v2]** `redact` iterates to a fixpoint, so it is idempotent by construction.
- Logging: JSON lines with ts, level, module, message, run_id, task_id. Redaction runs on every
  message and every extra field. `--verbose` sets INFO, `--debug` sets DEBUG.

### 6.2 models.py
- `ModelClient.complete(messages, tier, max_tokens) -> Completion(text, tokens_in, tokens_out)`.
  Tiers: small, large. Cost from config via `CostTable`.
- **[v2] Request line.** Every prompt ends with one machine-readable line
  `CRYSTALLIZER-REQUEST: {json}` (type `plan`, `action` or `summarize`, plus ids). Real models treat
  it as context; MockModel uses it to look up its script.
- MockModel: deterministic, seedable, scripted. Per step and per tier it can answer `correct`,
  `disagree` (sample k is correct only when `k % 3 == 0`; other samples are distinct wrong
  variants), `wrong` (all samples agree on the same wrong action), `error`, or `garbage`.
- AnthropicClient: lazy import of `anthropic`; clear error if missing. Reads `ANTHROPIC_API_KEY`
  from the environment only. Never logs request or response bodies. Model identifiers come from
  config, never hard-coded.

### 6.3 budget.py [v2]
- `BudgetGovernor.check()` runs before every model call; `charge(usage)` after. Limits are
  `max_cost_per_run`, `max_tokens_per_run`, `max_model_calls_per_run` (0 = off). The call that
  crosses a limit completes; the next is refused with exit code 7. On resume the governor is
  re-seeded from the run's recorded traces.

### 6.4 lock.py
- One run per workspace. Create `state_dir/run.lock` with `O_CREAT | O_EXCL`, write PID.
- If the lock exists and the PID is dead (or unreadable), treat as stale, remove, and retry once.
- Otherwise fail with exit code 5 and a clear message. Always release in `finally`.

### 6.5 db.py [v2] and memory.py
- One SQLite database `state_dir/state.db`: WAL mode, busy timeout, `synchronous=FULL`, versioned
  migrations. Tables: memory, journal, task_events, shadow_obs, live_runs.
- Memory entries: id, kind (decision|note|fact|task), text (redacted), tags, source_task,
  created_at, archived, summary_of.
- `query(text, k)`: score = fraction of query words present + `0.1 × 0.5^(age_days / half_life)`;
  ties break by id; deterministic.
- `summarize(older_than)`: compact old entries with the model into one note tagged `summary`, mark
  originals archived, never delete.

### 6.6 context.py
- `build(task, budget_tokens)` returns task, relevant memory, relevant files, recent trace digest,
  in deterministic order (section order fixed; within a section score descending, then key).
- **[v2] Budget invariant.** Each item costs `ceil((len(text) + 2) / chars_per_token)` (the 2 pays for
  the separator); the sum never exceeds the budget, hence neither does the rendered context.
- Relevant files: those named in the task (params, title, step descriptions), its acceptance
  commands, or its memory entries, each capped by `max_file_bytes`, read through the sandbox (so
  `state_dir` and escapes are unreachable).
- Over budget: drop lowest-score items first and record them. The task item is mandatory; if it
  alone exceeds the budget it is truncated and recorded **[v2]**. Every item is redacted.

### 6.7 planner.py
- `plan(goal)` asks the model (tier `planner_tier`) for JSON, extracts the first JSON object from
  the reply, validates it with pydantic, and returns a DAG of tasks: id, title, kind, params,
  depends_on, acceptance_commands, **steps [v2]**.
- Topological order is deterministic (ties by id). Duplicate ids, cycles and unknown dependencies
  are rejected.
- States: pending, running, done, failed, blocked. Every transition is persisted to `task_events`.
  Allowed transitions: pending→running|blocked, running→done|failed|pending,
  failed→pending, blocked→pending.
- A task is done only when all acceptance commands exit 0. A failed dependency blocks dependents.

### 6.8 journal.py
- Key: `(run_id, task_id, attempt, step_index, action_hash)` **[v2: attempt added so a deliberate
  retry re-executes while a resume never repeats]**.
- **[v2] Write-ahead protocol.** `started` is committed before the tool runs; `completed` with the
  redacted result after. On resume, `completed` → skip the tool and reuse the result; `started`
  only (in doubt) → re-execute if the action is idempotent (workspace file tools, `run_tests`,
  `pytest`, `mypy`, `git status|diff|add`), otherwise ask the human tier. `git commit` is
  reversible but not idempotent, so an in-doubt commit goes to the human.

### 6.9 checkpoint.py
- Atomic write: temp file, fsync, rename, fsync directory. SHA-256 over canonical JSON (all fields
  except `hash`). Keep the last 3.
- A corrupt latest checkpoint falls back to the previous; if none is valid, exit code 6.
- Save when a run starts, when a task attempt starts **[v2]**, and after each completed task.
  `resume` restarts from the first unfinished task and never repeats a completed side effect.

### 6.10 tools.py
- Tools: file_read, file_write, file_delete, shell, run_tests, **file_search, file_replace [v2]**
  (literal search, and literal replace with an optional identifier-boundary mode; they make
  renames genuinely mechanical).
- Every path is resolved and must stay inside the workspace. Reject `..` traversal and symlink
  escapes. Anything resolving inside `state_dir` is rejected for reads as well as writes **[v2]**,
  so the agent can never read or modify memory, checkpoints, traces, or skills. **[v2]** Writes and
  deletes inside `protected_paths` (default `.git`) are rejected, which blocks hook injection;
  writes open with `O_NOFOLLOW`.
- `shell` takes an argv list, never `shell=True`, never `sh -c`. The executable must be in
  `allowed_executables` by bare name. **[v2]** `python` resolves to the interpreter running
  crystallizer. Timeout on every command. Capture stdout, stderr, exit code. No stdin.
- The environment passed to subprocesses contains only `env_allowlist` variables. **[v2]** `PATH`
  keeps only absolute entries outside the workspace, so an agent-written file can never shadow an
  allowed executable.
- Output is redacted, truncated to `max_output_bytes`, and always treated as untrusted data.
- **[v2]** Tools live in a `ToolRegistry`; each has a `ToolSpec` (name, description, JSON Schema of
  its arguments, declared reversibility). Plugins may add tools; built-in names cannot be replaced.

### 6.11 policy.py
- `classify(action) -> PolicyDecision`. Reversible: file_read, file_search, file_write and
  file_replace inside the workspace, run_tests, and shell commands whose executable and subcommand
  are on the reversible allow-list (pytest, python -m pytest, ruff, mypy, git status, git diff,
  git add, git commit).
- Irreversible: file_delete, force-push, deploy, anything that spends money or uses credentials.
- **[v2] Strict argv parsing.** `git` with global options before the subcommand (`git -c …`) or with
  options that run programs or write elsewhere (`--output`, `--ext-diff`, `--exec-path`,
  `--upload-pack`, `-C`, …) is irreversible. `python` is reversible only as `python -m pytest`.
  `pytest` (also via `python -m pytest` and `run_tests`), `ruff` (`check`/`format` only) and `mypy`
  are reversible only with allow-listed options and workspace-relative positional arguments:
  `pytest --basetemp` (deletes a directory), `pytest -p PLUGIN` (loads code), `ruff --output-file`
  and `mypy --junit-xml` (write anywhere) are irreversible; `pytest -p no:NAME` is allowed.
- Unknown actions are irreversible (fail closed). Plugin tools are irreversible unless declared
  reversible and listed in `reversible_tools`.
- Irreversible actions require the human tier unless the action name (`file_delete`,
  `shell:git push`) or tool name is in `allow_irreversible`.

### 6.12 traces.py
- Append-only JSONL, one file per run, in `state_dir/traces/`. Flush and fsync per line.
- `normalize(step)`: keep constants as literals; the recorder stores raw redacted data; values that
  vary become typed slots in the miner, not in the recorder.
- **[v2]** If redaction changed any action argument, the step is flagged `redacted` and is never
  used for mining (a skill must never emit `[REDACTED]`). A torn final line (crash mid-write) is
  skipped on load.

### 6.13 skills/miner.py
- Input: verified traces. **[v2]** Only verified, successful, non-redacted, model- or agent-routed
  steps are used; skill-routed steps are excluded so skills never reinforce themselves.
- **[v2] Templating.** Each string argument is rewritten against the situation's param values
  (identity, `lower`, `upper`, `snake`; values of at least `min_template_length` characters; matches
  must sit on non-alphanumeric boundaries; longest value first; the template must render back to
  the exact original). The step signature is `(step_kind, tool, templated args)`.
- A step is mechanical iff its signature occurs in at least `min_repeats` distinct tasks: constant
  arguments and situation-derived arguments produce identical signatures; model-authored content
  does not. Non-mechanical steps break a sequence.
- Maximal runs of consecutive mechanical steps whose sequence of signatures occurs in at least
  `min_repeats` distinct tasks become candidate patterns, with provenance (trace ids) and the runs
  they came from. Deterministic ordering (support descending, then key).

### 6.14 skills/compiler.py
- Turn a pattern into a Skill JSON file: slots with types, steps with `{slot|filter}` templates,
  and a guard true only inside the observed domain: `eq`/`in` for categorical non-slot fields,
  `range` for numeric fields including `step_index` (observed min and max), `has_type` for slots,
  `exists` for fields present in every occurrence with mixed types.
- **[v2]** Skill id = `skill-` + first 12 hex of the SHA-256 of `(task_kind, steps, slots)`; the
  content hash covers the whole skill. Slot type inference order: int, identifier, path, string.
- Never generate Python. Write candidates to `state_dir/skills/candidates/`.

### 6.15 skills/executor.py
- Interprets a Skill: evaluate the guard against a Situation, bind slots from their sources,
  validate slot types, render templates, return a list of Actions. No eval, no exec, no imports of
  skill content.
- Rejects any skill with an unknown op, unknown filter, unknown slot, malformed or unresolved
  placeholder, or a slot source that is not a situation field. A missing field makes every
  condition false.

### 6.16 skills/shadow.py
- A candidate is evaluated only on occurrences NOT used to derive it: steps outside its provenance
  from runs outside its `derived_runs` (later runs). For each occurrence with a verified outcome,
  where the guard holds and the step kinds match, the executor's output is compared to the recorded
  verified (normalized) actions. Match counts as a pass.
- An unsafe-action diff is a skill-emitted action classified irreversible by policy where the
  verified action was reversible or absent. Any unsafe diff blocks promotion.
- Shadow evaluation runs beside the real path after each verified task; it never executes actions.
  **[v2]** Observations are keyed by `(skill, run, task, attempt, index)`, so online evaluation and
  the offline `skills evaluate` command are idempotent together.

### 6.17 skills/registry.py
- Layout under `state_dir/skills/`: `candidates/`, `active/`, `demoted/`, `manifest.json`.
- Only `registry.promote` may move a skill into `active/`. Promote requires
  `shadow_runs >= min_shadow_runs`, `pass_rate >= promote_pass_rate`, zero unsafe diffs.
- Auto-demote when the rolling pass rate over the last `demote_window` live runs (or all of them,
  while fewer exist) is below `demote_pass_rate`. **[v2]** A live skill whose emitted action is
  classified irreversible is demoted immediately. Demoted skills route back up the ladder; their
  failures become new traces.
- Every promotion and demotion is appended to the manifest history with its evidence
  (**[v2]** including the Wilson 95% lower bound of the pass rate, informational only).
- Manual `skills promote` uses the same checks. `--force` is allowed only with `--reason TEXT` and
  is logged in history.
- **[v2] Anti-flapping.** A pattern whose content hash equals a demoted version is never re-added;
  a changed pattern for a demoted id becomes a new version. Existing candidate or active ids are
  not churned by re-mining.
- **[v2] Import/export.** `skills export ID` prints the skill JSON; `skills import FILE` validates it
  with the executor and adds it as a *candidate* (origin `imported`), so it must pass local shadow
  testing before it can be promoted.

### 6.18 tiers.py [v2] and router.py
- Ladder: active skill (guard true) then small model then large model then human, configurable via
  `[router] ladder`. Every tier implements `Proposer.propose(StepRequest) -> TierResult`.
- Skill tier: active skills are tried in precedence order (most guard conditions, then most steps,
  then highest pass rate, then id); a skill applies only if the task's remaining planned step
  kinds match its step kinds and its guard is true.
- Model tiers: draw `samples` (small) or `large_samples` (large) completions; agreement is the
  fraction of samples equal to the modal normalized proposal (ties: the lexicographically smallest
  canonical form); accept only if agreement ≥ `agreement`. Confidence = agreement, **[v2]** capped at
  0.5 when fewer than 2 samples were drawn (a single sample cannot measure confidence).
- Escalate when: no skill or guard false, disagreement, unparseable output, a tier error,
  acceptance checks fail after `retries` retries, or the action is irreversible and confidence
  < `irreversible_confidence`. **[v2]** An irreversible action that is not allow-listed goes straight
  to the human tier.
- Human tier: approve or deny the pending proposal, or enter an action when no proposal exists. If
  stdin is not a TTY, fail closed (exit code 4).
- Every step logs and traces: route, escalation reason and path, tokens, cost, model calls.

### 6.19 runner.py
- For each ready task: build context, route each step, execute via tools through the journal,
  record the trace, run acceptance commands (through the same sandbox and policy **[v2]**), record
  the verdict, update state, run shadow evaluation and live-run accounting, save a checkpoint.
- Phase 2 direct path: every step goes to the small model with no router. Phase 4 replaces it with
  the router. Keep one runner; do not fork the code.

### 6.20 approval.py [v2]
- `Approver` protocol: `approve(request) -> bool` and `propose(request) -> Action | None`.
  `TTYApprover` (interactive; non-TTY raises exit 4), `DenyApprover`, `ScriptedApprover` (tests).

### 6.21 events.py and extensions.py [v2]
- Event bus with frozen, redacted events (section 9). Observer exceptions are logged and isolated.
- Registries for tools, tiers, model providers; `PluginContext`; plugin loading (section 9).

### 6.22 bench.py and scenario.py
- `crystallizer bench --scenario NAME --runs N [--profile e2e]` runs the scenario N times with
  MockModel, keeping skills between runs. **[v2]** Each run gets a fresh workspace with the scenario
  fixtures and a git repository; the state directory persists across runs; the plan comes from the
  scenario (validated as a DAG), so planning cost does not blur the comparison. Output JSON: per-run
  cost, tokens, model calls, route mix, skill counts by status, acceptance pass rate, plus summary
  (cost ratio last/first, first run with an active skill).

### 6.23 api.py [v2] and cli.py
- `Harness` is the stable programmatic API: `open` (config, profile, clock, seed, model override,
  approver, plugins, environment, dry run), `init`, `plan`, `adopt_plan` (use a plan produced by
  another planner or workflow engine), `run`, `resume`, `preview`, `status`, `report`,
  `skills_list`, `skill_show`, `skill_promote`, `skill_demote`, `skills_mine`, `skills_evaluate`,
  `skill_export`, `skill_import`, `tool_specs`, and `registry()`. The CLI is a thin layer over it.
- Commands: `init`, `plan GOAL`, `run`, `resume`, `status`, `skills list|show|promote|demote|mine|
  evaluate|export|import`, `report`, `bench`, `schemas export|check`, **`tools list` [v2]**.
- Global flags: `--config`, `--profile`, `--workspace`, `--verbose`, `--debug`, `--dry-run`, `--json`,
  accepted before or after the subcommand **[v2]** (so `bench ... --profile e2e` works as written).
- `--dry-run`: executes no tools, writes no state, makes no model calls, takes no lock; prints
  planned routing and actions, marking model steps as "would call small model".
- Exit codes: 0 success, 1 general error, 2 usage error, 3 a task failed, 4 human approval
  unavailable or denied, 5 workspace locked, 6 checkpoint unrecoverable, **7 budget exhausted [v2]**.
- `report`: cost, tokens and steps by route, skill counts, and the estimated saving against an
  all-large baseline (section 7.6).

## 7. Execution model [v2]

### 7.1 Steps and situations
A task has a planned skeleton `steps: [{kind, description}]`. The runner walks it; at position `i`
the situation is `{task_kind, step_kind = steps[i].kind, step_index = i, params}`. A tier returns
the action for step `i`; a skill may return actions for steps `i..i+n-1` when the planned kinds
match its step kinds. A task planned without steps runs in *open mode*: step kind `open`, the model
returns the next action or `{"done": true}`, capped at `max_open_steps`.

### 7.2 Run lifecycle
`plan GOAL` validates the plan and writes checkpoint 1 (no run). `run` refuses when an unfinished
run exists; otherwise it creates a run id, resets failed and blocked tasks to pending, marks the
cursor active and checkpoints. Tasks execute in deterministic topological order; a task whose
dependency failed or is blocked becomes blocked. At the end: mining, promotion, final checkpoint,
exit 3 if any task failed or is blocked. `resume` continues the active run with the same run id.

### 7.3 Task attempts and escalation floors
Attempts are numbered from 1. The *floor* is the lowest ladder tier allowed for this attempt
(0 = skill). If acceptance fails, the attempt is retried at the same floor up to `retries` times,
then the floor rises by one; when the floor passes the last model tier, the task fails. The cursor
(task, attempt, floor, failures at floor) is checkpointed at each attempt start, so resume re-enters
the same attempt and the journal skips its completed actions. Steps already executed in the run
are replayed from the trace, so a resume makes no model call for them either (redacted steps are
routed again, because their recorded action is not the executed one).

### 7.4 One step, end to end
context → ladder (from the floor) → proposal → policy classification (skills may not emit
irreversible actions; irreversible needs the human unless allow-listed; allow-listed needs
confidence ≥ threshold) → budget check before each model call → journal `started` → sandboxed tool
→ journal `completed` → trace step → event. A failed tool result ends the attempt.

### 7.5 After a task
Verdict record; memory entry; shadow observations for candidates; live-run records for skills
used (pass = acceptance passed), with immediate demotion when the rolling rate drops; checkpoint.

### 7.6 Baseline estimate
For `report`, each executed step is priced as one large-tier call. Model-routed steps use their own
recorded tokens divided by model calls; skill-routed steps use the mean per-call tokens of
model-routed steps in the same report (zero if none). The report labels this an estimate.

## 8. Security requirements
- Redaction is tested with fixtures for every pattern in section 6.1.
- Fail closed on ambiguity for irreversible actions; unknown actions are irreversible.
- The agent can never read or write `state_dir`. Only internal modules write there.
- Tool output is untrusted data. It is never executed and never treated as instructions.
- Skills are data. The executor is the only code that interprets them.
- **[v2]** Acceptance commands are model-authored and run through the same sandbox and policy.
- **[v2]** Plugins are imported only when named in `[plugins] enabled`; built-in names cannot be
  overridden; plugin tools default to irreversible.
- No network access in any test.

## 9. Extensibility and interoperability [v2]

### 9.1 Extension points
| Extension | Registered with | Selected by | Safety envelope |
|---|---|---|---|
| Ladder tier (external agent, rules engine, another model) | `add_tier(name, factory)` | `[router] ladder` | policy, confidence gate, budget, journal, sandbox, acceptance, traces |
| Model provider | `add_model_provider(name, factory)` | `[model] provider` | budget, cost table, redacted prompts |
| Tool | `add_tool(spec, handler)` | always available | sandbox helpers, irreversible unless trusted |
| Observer | `subscribe(handler)` | always | receives redacted, frozen events; cannot change decisions |

A plugin is an object with `name` and `register(context)`. It is supplied explicitly through the
Python API or discovered from the `crystallizer.plugins` entry-point group **only** when named in
`[plugins] enabled`.

### 9.2 Interchange formats
Plans, traces, verdicts, skills, manifests, checkpoints, events, tool descriptors and config have
committed JSON Schemas. Every CLI command supports `--json`. `tools list --json` exports tool
input schemas in the JSON Schema form used by function-calling APIs and MCP tool listings. Exit
codes are a stable contract.

### 9.3 Why this makes any agent system pluggable
An external system only has to map a `StepRequest` to actions. The harness supplies context,
planning, verification, crash safety, policy, budgets, and learning: once that system's verified
mechanical behavior repeats, it is crystallized into skills and stops costing anything. The
harness never trusts a tier more than its measured agreement and never lets any tier bypass
policy.

## 10. Bench scenarios

Each scenario is a TOML file in `scenarios/` with: name, description, seed, fixture files, a task
template instantiated per run with seeded parameters, and per-step scripts. Each step has: kind,
tool, args, `mechanical` (true or false), `tokens_in`, `tokens_out`, and `small_correct` (true
means the small tier's samples agree on the right action; false means they disagree, forcing
escalation to the large tier). Parameters vary per run using the seeded generator so that slots
genuinely vary; model-authored content is seeded random text that no template can derive.

1. `add-modules`: 4 modules per run. Per module: scaffold `src/{name}.py` from a template, scaffold
   `tests/test_{name}.py` from a template, write the function body (non-mechanical), `git add` the
   two files, and `git commit` with message `add {name}`. **[v2 amendment]** v1 specified 3
   mechanical steps (scaffold, scaffold, commit); git cannot commit untracked files from a single
   argv without staging, so staging is its own mechanical step: 5 steps per module, 20 per run,
   16 mechanical. The commit step sets `small_correct = false` (4 per run).
2. `fix-tests`: 4 failing tests per run, each with one non-mechanical fix step followed by a
   mechanical rerun-and-commit sequence (run the test, `git add`, `git commit`): 16 steps per run,
   12 mechanical. The commit step sets `small_correct = false`.
3. `rename-refactor`: rename 4 symbols per run; per symbol 3 mechanical steps (`file_search`,
   `file_replace`, `run_tests`) and 1 non-mechanical review note: 16 steps per run, 12 mechanical.
   The search step sets `small_correct = false`.

Under the e2e profile each scenario mines a skill after run 1, shadow-tests it in runs 2 and 3, and
has it active by run 4.

## 11. Testing requirements
- Unit tests for every module, using MockModel.
- Failure injection: crash mid-task then resume with no duplicate side effects (at every fault
  point); corrupt latest checkpoint falls back; over-budget context; bad skill demoted; irreversible
  action forced to human; unknown action treated as irreversible; path traversal blocked; symlink
  escape blocked; write into `state_dir` blocked; secret redacted in logs, traces, and errors;
  disagreement escalates; second concurrent run fails with exit code 5; stale lock recovered;
  **[v2]** budget exhaustion exits 7 and resumes; observer exceptions isolated.
- Property tests (hypothesis, derandomized): a guard never matches inputs outside the mined domain;
  the context builder never exceeds its budget; checkpoints round-trip exactly; executor output for
  a valid skill always type-checks; redaction never raises and is idempotent.
- Skill safety tests: an unknown guard op, unknown filter, or unresolved placeholder is rejected;
  no test may pass a skill through eval, exec, or import.
- End-to-end (`tests/e2e`): run `add-modules` 5 times under the e2e profile. Assert: at least one
  skill promoted by run 4; the cost of run 5 is at most 50% of run 1; acceptance checks pass in
  every run; the trace log of run 5 contains skill-routed steps.
- Coverage: at least 90% line coverage on `src/crystallizer`, enforced by `make check`.

## 12. Makefile targets
- `install`: create the environment and install with dev extras.
- `lint`: `ruff check` and `ruff format --check`.
- `typecheck`: `mypy --strict src tests`.
- `test`: `pytest --cov=src/crystallizer --cov-fail-under=90`.
- `schemas`: regenerate `/schemas`.
- `check`: lint, typecheck, test, and the schema drift check.

## 13. Phases and acceptance
1. **Foundation**: packaging, CI, LICENSE, docs set, errors, clock, hashing, faults, schemas,
   config, redaction, logging, models (MockModel), budget, db, lock, memory, context, journal,
   checkpoint, tools, policy, events, extensions, tier protocol. Accept: `make check` green, all
   failure-injection tests for these modules present.
2. **Planning and traces**: planner, traces, approval, runner (direct path), api, CLI `init`,
   `plan`, `run`, `resume`, `status`, `schemas`, `tools`. Accept: a scripted MockModel project runs,
   crashes, resumes.
3. **Skills**: miner, compiler, executor, shadow, registry, CLI `skills`. Accept: mining, compiling,
   shadow-testing, promotion, and demotion work on recorded traces.
4. **Router**: built-in tiers, full ladder, escalation, human tier, budget enforcement, cost
   accounting, `report`, shadow evaluation inside the runner. Accept: routing decisions and
   escalations are logged and tested.
5. **Bench and docs**: three scenarios, `bench`, e2e test, AnthropicClient, README (quickstart,
   architecture mermaid diagram, exit codes, license sentence, honest limits), CONTRIBUTING,
   NOTES.md. Accept: the definition of done.

## 14. Definition of done
- `make check` passes with no warnings on Python 3.11 and 3.12 (CI green).
- README quickstart works from a clean clone using MockModel only.
- Schemas are generated and committed, and the drift check passes.
- The e2e test and `crystallizer bench --scenario add-modules --runs 5 --profile e2e` show the cost
  drop.
- LICENSE matches section 3 exactly, and no open-source license text exists anywhere.
- NOTES.md contains: the honest limits from section 1, a Decisions list, a Deviations list, and
  known risks.

---

## Appendix A. Changes from v1.0

| Area | v1.0 | v2.0 | Reason |
|---|---|---|---|
| Tiers | fixed ladder | `Proposer` protocol, configurable ladder, plugin tiers | any agent system can plug in under the same envelope |
| Extensibility | none | plugin kernel, event bus, Harness API, tool descriptors | embed in other workflows |
| Budgets | none | hard per-run limits, exit 7 | prevent runaway cost |
| Step model | undefined | planned step skeleton, open mode, `step_index` | situations must be known before routing |
| Slots | name → type | `{type, source}` | unambiguous binding |
| Journal key | no attempt | attempt included; write-ahead with in-doubt handling | retries vs resumes; crash consistency |
| Trace verification | mutable flag | append-only verdict records | keep traces append-only |
| Redaction | raw patterns | boundaries, key components, fixpoint | avoid false positives, guarantee idempotency |
| Sandbox | paths, exes, env, time | + state_dir reads, protected `.git`, O_NOFOLLOW, strict git/python parsing | close hook-injection and option-injection holes |
| Tools | 5 | + file_search, file_replace | mechanical renames |
| Acceptance | ran directly | through sandbox and policy | model-authored commands |
| Confidence | undefined for one sample | capped at 0.5 below 2 samples | fail closed |
| Skills lifecycle | promote/demote | + anti-flapping, immediate unsafe demotion, import/export, Wilson bound | stability and portability |
| add-modules | 3 mechanical/module | 4 mechanical/module | git staging needs its own argv |
| License holder | Logic-Love | Koosha KZ | repository owner's decision |

## Appendix B. Glossary
- **Mechanical step**: every argument is a constant or derivable from the situation.
- **Occurrence**: one place in the verified traces where a pattern's step sequence appears.
- **Floor**: the lowest ladder tier allowed for a task attempt.
- **In doubt**: an action journaled as started whose completion was never recorded.
- **Unsafe diff**: a skill-emitted irreversible action where the verified action was reversible
  or absent.
