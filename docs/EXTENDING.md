# Extending crystallizer

crystallizer is built to host other agent systems, not to replace them. Anything that can turn a
step request into actions can become a rung of the cost ladder, and it immediately inherits
context building, crash safety, policy, budgets, verification and skill crystallization.

## 1. Plugin shape

```python
from crystallizer.extensions import PluginContext


class MyPlugin:
    name = "my_plugin"

    def register(self, context: PluginContext) -> None:
        context.add_tier("my_agent", MyAgentTier)          # a ladder tier
        context.add_model_provider("my_llm", make_client)  # a model provider
        context.add_tool(spec, handler)                     # a tool
        context.subscribe(on_event)                         # an observer
```

Two ways to load it:

1. **Embedding**: pass it to the Python API.
   `Harness.open(workspace, plugins=[MyPlugin()])`
2. **Installed package**: expose an entry point and name it in config.

   ```toml
   # the plugin's pyproject.toml
   [project.entry-points."crystallizer.plugins"]
   my_plugin = "my_package.plugin:MyPlugin"
   ```

   ```toml
   # crystallizer.toml
   [plugins]
   enabled = ["my_plugin"]
   ```

   Nothing is imported unless it is named in `enabled`; an unknown name is a usage error (exit 2).

## 2. Adding an agent system as a ladder tier

A tier implements the `Proposer` protocol from `crystallizer.tiers`:

```python
from crystallizer.schemas import Action, Proposal, StepRequest, TierResult, Usage
from crystallizer.tiers import TierContext


class MyAgentTier:
    def __init__(self, context: TierContext) -> None:
        self.context = context  # config, model client, costs, budget, policy, bus, tool specs

    @property
    def name(self) -> str:
        return "my_agent"

    def propose(self, request: StepRequest) -> TierResult:
        # request.situation: task_kind, step_kind, step_index, params
        # request.context: budgeted, redacted context text
        # request.task: the task, its planned steps and acceptance commands
        action = Action(tool="file_write", args={"path": "notes.md", "content": "..."})
        return TierResult(
            tier=self.name,
            proposal=Proposal(tier=self.name, actions=[action], confidence=0.8),
            usage=Usage(tokens_in=1200, tokens_out=300, cost=0.004, model_calls=1),
        )
```

Place it in the ladder (skill first and human last are enforced):

```toml
[router]
ladder = ["skill", "small", "my_agent", "large", "human"]
```

What the harness guarantees around your tier:

- **Policy**: every proposed action is classified; irreversible actions need the human tier unless
  allow-listed, and allow-listed ones need confidence ≥ `irreversible_confidence`.
- **Budget**: report honest `usage`; it is charged to the run and enforced.
- **Crash safety**: actions run through the write-ahead journal; resume never repeats them.
- **Verification**: the task's acceptance commands decide success, not the tier.
- **Learning**: verified mechanical steps your tier produced are mined; once promoted, the skill
  tier answers them and your tier is no longer called for them.
- **Escalation**: return `TierResult(tier=..., escalation=EscalationReason.DISAGREEMENT, ...)` (or
  another reason) to pass the step up the ladder.

## 3. Adding a model provider

```python
from crystallizer.config import Config
from crystallizer.schemas import Completion, Message


class MyClient:
    def complete(self, messages: list[Message], tier: str, max_tokens: int) -> Completion:
        ...  # call your provider with the identifier for `tier` from config


def make_client(config: Config) -> MyClient:
    return MyClient()
```

Select it with `[model] provider = "my_llm"`. The prompt's last line is
`CRYSTALLIZER-REQUEST: {json}`; answer action requests with JSON `{"tool": ..., "args": {...}}` or
`{"done": true}`, and plan requests with a plan object (see `schemas/plan.schema.json`).

## 4. Adding a tool

```python
from crystallizer.schemas import ToolResult, ToolSpec

spec = ToolSpec(
    name="http_get",
    description="Fetch a URL.",
    args_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    reversible=True,
)


def handler(args, sandbox) -> ToolResult:
    # use sandbox.resolve(path, write=...) for any file access
    return ToolResult(ok=True, output="...")
```

Plugin tools are **irreversible** (human approval) unless they declare `reversible=True` **and** are
listed in `[policy] reversible_tools`. Built-in tool names cannot be replaced.

## 5. Observing events

Handlers receive frozen, redacted `Event` objects (`run.started`, `task.started`, `step.routed`,
`step.executed`, `escalation`, `skill.mined`, `skill.promoted`, `skill.demoted`,
`checkpoint.saved`, `budget.exhausted`, `task.finished`, `run.finished`). A handler that raises is
logged and ignored. Handlers cannot influence routing or execution.

## 6. Integrating with other workflows

- Every CLI command accepts `--json`; exit codes are a stable contract (see README).
- JSON Schemas for plans, traces, skills, manifests, checkpoints, events, tool descriptors and
  config are committed under `schemas/`.
- `crystallizer tools list --json` exports tool input schemas in JSON Schema form, the format used
  by function-calling APIs and MCP tool listings.
- Skills move between workspaces with `skills export` / `skills import`; imports start as
  candidates and must pass local shadow testing.
