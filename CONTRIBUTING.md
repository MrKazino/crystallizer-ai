# Contributing

This repository is proprietary (see LICENSE). Contributions are accepted only from people who have
written permission from the copyright holder.

## Setup

    make install        # creates .venv and installs the package with dev extras
    make check          # lint, strict typing, tests with >= 90% coverage, schema drift check

Python 3.11 or 3.12. No network access or API keys are needed for any test.

## Rules

- Read `docs/SPEC.md` first; it is the contract. Record ambiguities you resolve in `NOTES.md`
  under Decisions, and anything you cannot meet under Deviations.
- Complete code only: no stubs, TODOs, placeholders or `pass` bodies.
- Type hints and docstrings on every public module, class and function; `mypy --strict` clean.
- Tests are deterministic: inject `Clock` and seeded RNGs, use `MockModel`, never the network.
- Never execute, evaluate or import generated code. Skills are data.
- Never log, trace, persist or print secrets; route strings through `crystallizer.redaction`.
- When a data model changes, run `make schemas` and commit the regenerated `schemas/`.
- Never claim a performance or cost improvement that a test or benchmark does not measure.

## Commit and review

One phase or one focused change per commit, with a message that says what and why. `make check`
must pass before pushing; CI runs it on Python 3.11 and 3.12.
