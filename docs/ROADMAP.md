# Roadmap

Each phase ends with `make check` green (lint, strict typing, ≥ 90% coverage, schema drift
check), a commit, and a summary. Status is updated as phases land.

## Phase 1: Foundation (done)
- [x] Packaging (pinned deps, extras), Makefile, CI on 3.11 and 3.12, LICENSE, .gitignore
- [x] Docs: SPEC v2.0, ARCHITECTURE, EXTENDING, ROADMAP, README, NOTES, CONTRIBUTING
- [x] errors, clock, hashing, faults
- [x] schemas with export and drift check; config with strict keys and profiles
- [x] redaction (linear, idempotent) and JSON logging
- [x] models: ModelClient protocol, MockModel, cost table; budget governor
- [x] db, lock, memory, context, journal, checkpoint
- [x] tools (sandbox, registry), policy, events, extensions, tier protocol
- [x] Failure-injection tests: stale lock, second run exit 5, corrupt checkpoint fallback, exit 6,
      in-doubt journal entries, traversal, symlink escape, state_dir writes, redaction everywhere

## Phase 2: Planning and traces (done)
- [x] planner (JSON extraction, DAG validation, deterministic topological order, transitions)
- [x] traces (append-only JSONL, verdicts, redaction flag, torn-line tolerance)
- [x] approval (TTY, deny, scripted)
- [x] runner direct path (small model), api Harness, CLI init/plan/run/resume/status/tools
- [x] Crash at every fault point and resume with no duplicate side effects; resume replays
      executed steps without model calls

## Phase 3: Skills (done)
- [x] miner, compiler, executor, shadow, registry
- [x] CLI skills list/show/promote/demote/mine/evaluate/export/import
- [x] Property tests: guard domain, executor output type-checks; skill safety tests (including a
      static check that the skills package never calls eval/exec/compile or imports importlib)

## Phase 4: Router (done)
- [x] built-in tiers, configurable ladder, plugin tiers
- [x] escalation reasons and paths, human tier, confidence gate, budget enforcement
- [x] shadow and live accounting in the runner, demotion, report with baseline estimate
- [x] overhead records keep cost accounting exact across failed routing and interruptions

## Phase 5: Bench and docs (done)
- [x] scenario loader, three scenarios, bench command
- [x] e2e test (skill by run 4, run 5 ≤ 50% of run 1 cost, acceptance 100%)
- [x] AnthropicClient (lazy import, fake-client tested)
- [x] README quickstart verified from a clean clone; NOTES final

## Measured benchmark results (MockModel, `--profile e2e`, 5 runs)

| Scenario | Run 1 cost | Run 5 cost | Ratio | First run with skills | Acceptance |
|---|---|---|---|---|---|
| add-modules | 0.042 | 0.0096 | 0.229 | 4 | 100% |
| fix-tests | 0.03795 | 0.0096 | 0.253 | 4 | 100% |
| rename-refactor | 0.03795 | 0.0096 | 0.253 | 4 | 100% |

Costs are illustrative units from scripted token counts; they show the mechanics, not real-model
savings.

## Possible next steps (not in scope of v2.0)
- Parallel execution of independent tasks within the lock.
- A plan cache keyed by goal shape.
- Optional statistical promotion gates (the Wilson bound is already recorded as evidence).
