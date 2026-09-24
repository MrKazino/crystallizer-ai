"""Action classification and the irreversible-action policy (fail closed).

Reversible: ``file_read``, ``file_search``, ``file_write`` and ``file_replace`` inside the
workspace, ``run_tests``, and ``shell`` commands whose executable and subcommand are on the
reversible allow-list (``pytest``, ``python -m pytest``, ``ruff``, ``mypy``, ``git status``,
``git diff``, ``git add``, ``git commit``). Everything else is irreversible: ``file_delete``,
force pushes, deploys, anything that spends money or uses credentials, and every unknown action.

``git`` commands are parsed strictly: global options before the subcommand (``git -c ...``) and
options that run programs or write outside the workspace (``--output``, ``--ext-diff``,
``--exec-path`` ...) make the command irreversible. ``python`` is reversible only as
``python -m pytest``.

``pytest`` (also via ``python -m pytest`` and the ``run_tests`` tool), ``ruff`` and ``mypy`` are
reversible only with allow-listed options and workspace-relative positional arguments: options
such as ``pytest --basetemp`` (deletes a directory), ``pytest -p PLUGIN`` (loads code),
``ruff --output-file`` or ``mypy --junit-xml`` (write anywhere) make the command irreversible.
``pytest -p no:NAME`` (disabling a plugin) is allowed. ``ruff`` must be ``ruff check`` or
``ruff format``.

An irreversible action requires the human tier unless its action name (``file_delete``,
``shell:git push`` ...) or tool name is in ``[policy] allow_irreversible``. Plugin tools are
irreversible unless they declare themselves reversible **and** are listed in
``[policy] reversible_tools``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

from crystallizer.config import PolicyConfig
from crystallizer.schemas import Action, PolicyDecision, Reversibility, ToolSpec

REVERSIBLE_TOOLS = frozenset(
    {"file_read", "file_search", "file_write", "file_replace", "run_tests"}
)
REVERSIBLE_GIT = frozenset({"status", "diff", "add", "commit"})
DANGEROUS_GIT_OPTIONS = (
    "--output",
    "--ext-diff",
    "--open-files-in-pager",
    "-O",
    "--exec-path",
    "--upload-pack",
    "--receive-pack",
    "--git-dir",
    "--work-tree",
    "--config-env",
    "-c",
    "-C",
)
FORCE_OPTIONS = frozenset({"-f", "--force", "--force-with-lease", "--force-if-includes"})
PYTEST_FLAGS = frozenset(
    {
        "-q", "-qq", "-v", "-vv", "-x", "-s", "-l", "-ra", "-rA", "-rf",
        "--quiet", "--verbose", "--exitfirst", "--strict-markers", "--no-header",
        "--collect-only", "--co", "--lf", "--ff", "--last-failed", "--failed-first",
        "--disable-warnings", "--showlocals",
    }
)  # fmt: skip
PYTEST_VALUE_OPTIONS = frozenset({"-k", "-m", "--tb", "--maxfail", "--durations", "-W"})
RUFF_SUBCOMMANDS = frozenset({"check", "format"})
RUFF_FLAGS = frozenset(
    {"--fix", "--no-fix", "--diff", "--check", "-q", "--quiet", "--statistics", "--show-fixes"}
)
RUFF_VALUE_OPTIONS = frozenset(
    {"--select", "--ignore", "--extend-select", "--line-length", "--target-version"}
)
MYPY_FLAGS = frozenset(
    {
        "--strict", "--ignore-missing-imports", "--no-incremental", "--pretty",
        "--no-error-summary", "--show-error-codes", "--check-untyped-defs",
        "--warn-unused-ignores", "--no-color-output",
    }
)  # fmt: skip
MYPY_VALUE_OPTIONS = frozenset(
    {"-p", "-m", "--package", "--module", "--python-version", "--follow-imports"}
)
IDEMPOTENT_TOOLS = frozenset(
    {"file_read", "file_search", "file_write", "file_replace", "run_tests"}
)
IDEMPOTENT_SHELL = frozenset(
    {
        "shell:pytest",
        "shell:python -m pytest",
        "shell:mypy",
        "shell:git status",
        "shell:git diff",
        "shell:git add",
    }
)


def shell_action_name(argv: list[str]) -> str:
    """Return the policy name of a shell command, e.g. ``shell:git commit``."""
    if not argv:
        return "shell"
    executable = argv[0]
    if executable == "git" and len(argv) > 1 and not argv[1].startswith("-"):
        return f"shell:git {argv[1]}"
    if executable == "python" and argv[1:3] == ["-m", "pytest"]:
        return "shell:python -m pytest"
    return f"shell:{executable}"


def _safe_positional(argument: str) -> bool:
    path = PurePosixPath(argument.split("::", 1)[0])
    return (
        bool(argument)
        and "\x00" not in argument
        and not path.is_absolute()
        and (".." not in path.parts)
    )


def arguments_allowed(
    args: Sequence[str],
    flags: frozenset[str],
    value_options: frozenset[str],
    *,
    pytest_plugins: bool = False,
) -> bool:
    """True if every option is allow-listed and every positional is workspace-relative."""
    index = 0
    while index < len(args):
        argument = args[index]
        if pytest_plugins and argument == "-p":
            if index + 1 < len(args) and args[index + 1].startswith("no:"):
                index += 2
                continue
            return False
        if pytest_plugins and argument.startswith("-p") and argument[2:].startswith("no:"):
            index += 1
            continue
        if argument.startswith("-"):
            name, has_value, _ = argument.partition("=")
            if argument in flags:
                index += 1
                continue
            if name in value_options:
                if not has_value and index + 1 >= len(args):
                    return False
                index += 1 if has_value else 2
                continue
            return False
        if not _safe_positional(argument):
            return False
        index += 1
    return True


def pytest_arguments_allowed(args: Sequence[str]) -> bool:
    """Allow-list check for pytest arguments (``run_tests`` and pytest commands)."""
    return arguments_allowed(args, PYTEST_FLAGS, PYTEST_VALUE_OPTIONS, pytest_plugins=True)


def _dangerous_git_option(argument: str) -> bool:
    return any(argument == opt or argument.startswith(opt + "=") for opt in DANGEROUS_GIT_OPTIONS)


class Policy:
    """Classifies actions and decides whether a human must approve them."""

    def __init__(
        self, config: PolicyConfig, plugin_tools: dict[str, ToolSpec] | None = None
    ) -> None:
        """Create a policy; ``plugin_tools`` are the non-built-in tool descriptors."""
        self._allow = frozenset(config.allow_irreversible)
        self._trusted = frozenset(config.reversible_tools)
        self._plugin_tools = dict(plugin_tools or {})

    def classify(self, action: Action) -> PolicyDecision:
        """Classify ``action`` and mark whether it requires the human tier."""
        name, classification, reason = self._classify(action)
        requires_human = classification is Reversibility.IRREVERSIBLE and not (
            name in self._allow or action.tool in self._allow
        )
        return PolicyDecision(
            action_name=name,
            classification=classification,
            reason=reason,
            requires_human=requires_human,
        )

    def is_idempotent(self, action: Action) -> bool:
        """True if running ``action`` twice has the same effect as once (safe to redo in doubt).

        Only reversible actions qualify: workspace file tools, ``run_tests``, and read-only or
        staging shell commands. ``git commit`` is reversible but not idempotent.
        """
        decision = self.classify(action)
        if decision.irreversible:
            return False
        return action.tool in IDEMPOTENT_TOOLS or decision.action_name in IDEMPOTENT_SHELL

    def _classify(self, action: Action) -> tuple[str, Reversibility, str]:
        tool = action.tool
        if tool == "run_tests":
            extra = action.args.get("args", [])
            if not isinstance(extra, list) or not pytest_arguments_allowed(extra):
                return tool, Reversibility.IRREVERSIBLE, "pytest option is not on the allow-list"
            return tool, Reversibility.REVERSIBLE, "workspace-confined tool"
        if tool in REVERSIBLE_TOOLS:
            return tool, Reversibility.REVERSIBLE, "workspace-confined tool"
        if tool == "file_delete":
            return tool, Reversibility.IRREVERSIBLE, "deletion cannot be undone"
        if tool == "shell":
            argv = action.args.get("argv")
            if not isinstance(argv, list) or not argv:
                return "shell", Reversibility.IRREVERSIBLE, "malformed shell command"
            return self._classify_shell(argv)
        spec = self._plugin_tools.get(tool)
        if spec is not None:
            if spec.reversible and tool in self._trusted:
                return tool, Reversibility.REVERSIBLE, "plugin tool trusted as reversible"
            return tool, Reversibility.IRREVERSIBLE, "plugin tool not trusted as reversible"
        return tool, Reversibility.IRREVERSIBLE, "unknown action (fail closed)"

    def _classify_shell(self, argv: list[str]) -> tuple[str, Reversibility, str]:
        name = shell_action_name(argv)
        executable = argv[0]
        irreversible = Reversibility.IRREVERSIBLE
        option_reason = f"{executable} option or path is not on the allow-list"
        if executable == "pytest":
            if not pytest_arguments_allowed(argv[1:]):
                return name, irreversible, option_reason
            return name, Reversibility.REVERSIBLE, "pytest is on the reversible allow-list"
        if executable == "ruff":
            if len(argv) < 2 or argv[1] not in RUFF_SUBCOMMANDS:
                return name, irreversible, "only ruff check and ruff format are reversible"
            if not arguments_allowed(argv[2:], RUFF_FLAGS, RUFF_VALUE_OPTIONS):
                return name, irreversible, option_reason
            return name, Reversibility.REVERSIBLE, "ruff is on the reversible allow-list"
        if executable == "mypy":
            if not arguments_allowed(argv[1:], MYPY_FLAGS, MYPY_VALUE_OPTIONS):
                return name, irreversible, option_reason
            return name, Reversibility.REVERSIBLE, "mypy is on the reversible allow-list"
        if executable == "python":
            if argv[1:3] == ["-m", "pytest"]:
                if not pytest_arguments_allowed(argv[3:]):
                    return name, irreversible, "pytest option or path is not on the allow-list"
                return name, Reversibility.REVERSIBLE, "python -m pytest is reversible"
            return name, irreversible, "python may run arbitrary code"
        if executable == "git":
            if len(argv) < 2 or argv[1].startswith("-"):
                return name, irreversible, "git global options are not allowed"
            subcommand = argv[1]
            if subcommand == "push" and any(
                arg in FORCE_OPTIONS or arg.startswith("+") for arg in argv[2:]
            ):
                return name, irreversible, "force-push rewrites remote history"
            if subcommand not in REVERSIBLE_GIT:
                return name, irreversible, f"git {subcommand} is not on the reversible allow-list"
            if any(_dangerous_git_option(arg) for arg in argv[2:]):
                return name, irreversible, "git option can run programs or write outside"
            return name, Reversibility.REVERSIBLE, f"git {subcommand} is reversible"
        return name, irreversible, f"{executable} is not on the reversible allow-list"
