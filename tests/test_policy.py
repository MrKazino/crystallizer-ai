"""Policy classification: reversible allow-list, fail-closed defaults, allow-listing."""

from __future__ import annotations

import pytest

from crystallizer.config import PolicyConfig
from crystallizer.policy import Policy, shell_action_name
from crystallizer.schemas import Action, Reversibility, ToolSpec


def shell(*argv: str) -> Action:
    return Action(tool="shell", args={"argv": list(argv)})


@pytest.mark.parametrize(
    "action",
    [
        Action(tool="file_read", args={"path": "a"}),
        Action(tool="file_write", args={"path": "a", "content": "b"}),
        Action(tool="file_search", args={"query": "a"}),
        Action(tool="file_replace", args={"glob": "*", "old": "a", "new": "b"}),
        Action(tool="run_tests"),
        shell("pytest", "-q"),
        shell("python", "-m", "pytest", "-q"),
        shell("ruff", "check", "."),
        shell("mypy", "src"),
        shell("git", "status"),
        shell("git", "diff", "--stat"),
        shell("git", "add", "src/a.py"),
        shell("git", "commit", "-m", "add a"),
    ],
)
def test_reversible(action: Action) -> None:
    decision = Policy(PolicyConfig()).classify(action)
    assert decision.classification is Reversibility.REVERSIBLE
    assert not decision.requires_human
    assert not decision.irreversible


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (Action(tool="file_delete", args={"path": "a"}), "deletion"),
        (shell("git", "push", "--force"), "force-push"),
        (shell("git", "push", "origin", "+main"), "force-push"),
        (shell("git", "push"), "not on the reversible"),
        (shell("git", "-c", "alias.x=!rm", "x"), "global options"),
        (shell("git"), "global options"),
        (shell("git", "diff", "--output=/etc/x"), "run programs"),
        (shell("git", "diff", "--ext-diff"), "run programs"),
        (shell("python", "-c", "import os"), "arbitrary code"),
        (shell("python", "script.py"), "arbitrary code"),
        (shell("curl", "https://x"), "not on the reversible"),
        (Action(tool="shell", args={"argv": []}), "malformed"),
        (Action(tool="shell", args={"argv": "git status"}), "malformed"),
        (Action(tool="deploy", args={}), "unknown action"),
        (Action(tool="spend_money", args={"amount": 5}), "unknown action"),
    ],
)
def test_irreversible_and_fail_closed(action: Action, reason: str) -> None:
    decision = Policy(PolicyConfig()).classify(action)
    assert decision.classification is Reversibility.IRREVERSIBLE
    assert decision.requires_human
    assert reason in decision.reason


def test_allow_irreversible_by_action_or_tool_name() -> None:
    policy = Policy(PolicyConfig(allow_irreversible=["file_delete", "shell:git push"]))
    assert not policy.classify(Action(tool="file_delete", args={"path": "a"})).requires_human
    push = policy.classify(shell("git", "push"))
    assert push.irreversible
    assert not push.requires_human
    assert policy.classify(shell("git", "rebase")).requires_human


def test_plugin_tools_need_declaration_and_trust() -> None:
    spec = ToolSpec(name="lookup", description="d", args_schema={}, reversible=True, builtin=False)
    untrusted = Policy(PolicyConfig(), {"lookup": spec})
    assert untrusted.classify(Action(tool="lookup")).irreversible
    trusted = Policy(PolicyConfig(reversible_tools=["lookup"]), {"lookup": spec})
    assert not trusted.classify(Action(tool="lookup")).irreversible
    undeclared = spec.model_copy(update={"reversible": False})
    assert (
        Policy(PolicyConfig(reversible_tools=["lookup"]), {"lookup": undeclared})
        .classify(Action(tool="lookup"))
        .irreversible
    )


def test_shell_action_names() -> None:
    assert shell_action_name([]) == "shell"
    assert shell_action_name(["git", "commit", "-m", "x"]) == "shell:git commit"
    assert shell_action_name(["git", "-c", "x"]) == "shell:git"
    assert shell_action_name(["python", "-m", "pytest"]) == "shell:python -m pytest"
    assert shell_action_name(["python", "-c", "x"]) == "shell:python"


@pytest.mark.parametrize(
    "action",
    [
        shell("pytest", "--basetemp=/home/user"),
        shell("pytest", "--basetemp", "/tmp/x"),
        shell("pytest", "-p", "evil_plugin"),
        shell("pytest", "-pevil"),
        shell("pytest", "--junitxml=/etc/report.xml"),
        shell("pytest", "/etc/passwd"),
        shell("pytest", "../outside/test_x.py"),
        shell("pytest", "-k"),
        shell("python", "-m", "pytest", "--rootdir=/"),
        shell("ruff", "check", "--output-file", "/etc/x"),
        shell("ruff", "clean"),
        shell("ruff"),
        shell("mypy", "--junit-xml", "/tmp/j.xml", "src"),
        shell("mypy", "--python-executable", "/bin/sh"),
        Action(tool="run_tests", args={"args": ["--basetemp=/"]}),
        Action(tool="run_tests", args={"args": "-q"}),
    ],
)
def test_dangerous_tool_options_are_irreversible(action: Action) -> None:
    decision = Policy(PolicyConfig()).classify(action)
    assert decision.irreversible
    assert decision.requires_human


@pytest.mark.parametrize(
    "action",
    [
        shell("pytest", "-q", "-p", "no:cacheprovider", "tests/test_a.py::test_one"),
        shell("pytest", "-pno:cacheprovider", "-k", "fast and not slow", "--tb=short"),
        shell("python", "-m", "pytest", "-x", "--maxfail=1", "tests"),
        shell("ruff", "format", "--check", "src"),
        shell("ruff", "check", "--select", "E,F", "--fix", "."),
        shell("mypy", "--strict", "-p", "crystallizer"),
        Action(
            tool="run_tests", args={"args": ["-q", "-p", "no:cacheprovider", "tests/test_a.py"]}
        ),
    ],
)
def test_plain_tool_usage_stays_reversible(action: Action) -> None:
    assert not Policy(PolicyConfig()).classify(action).irreversible


def test_idempotency() -> None:
    policy = Policy(PolicyConfig())
    assert policy.is_idempotent(shell("git", "add", "a.py"))
    assert policy.is_idempotent(Action(tool="file_write", args={"path": "a", "content": "b"}))
    assert not policy.is_idempotent(shell("git", "commit", "-m", "x"))
    assert not policy.is_idempotent(Action(tool="file_delete", args={"path": "a"}))
