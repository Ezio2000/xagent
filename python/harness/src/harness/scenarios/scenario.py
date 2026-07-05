"""High-level scenario assembly for controlled runtime tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kernel import (
    AgentEvent,
    AgentLoop,
    AgentResult,
    ApprovalPolicy,
    LoopLimits,
    Message,
    ModelClient,
    ModelOptions,
    ResponseFormat,
    RunController,
    RunJournal,
    RunStore,
    RuntimeContext,
    RuntimeHook,
    ToolChoice,
    ToolRegistryProtocol,
    ToolSchedulerFactory,
)

from harness.observation.events import collect_events


class KernelScenario:
    """Reusable harness wrapper that assembles and runs an `AgentLoop`."""

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistryProtocol | None = None,
        limits: LoopLimits | None = None,
        model_options: ModelOptions | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        hooks: Sequence[RuntimeHook] | None = None,
        approval_metadata: Mapping[str, Any] | None = None,
        approval_policy: ApprovalPolicy | None = None,
        run_store: RunStore | None = None,
        run_journal: RunJournal | None = None,
        trace: bool = True,
        tool_scheduler_factory: ToolSchedulerFactory | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.limits = limits
        self.model_options = model_options
        self.tool_choice = tool_choice
        self.response_format = response_format
        self.hooks = tuple(hooks or ())
        self.approval_metadata = dict(approval_metadata or {})
        self.approval_policy = approval_policy
        self.run_store = run_store
        self.run_journal = run_journal
        self.trace = trace
        self.tool_scheduler_factory = tool_scheduler_factory

    def agent(self) -> AgentLoop:
        """Create a fresh `AgentLoop` using this scenario's configured parts."""

        return AgentLoop(
            model=self.model,
            tools=self.tools,
            limits=self.limits,
            model_options=self.model_options,
            tool_choice=self.tool_choice,
            response_format=self.response_format,
            hooks=self.hooks,
            approval_metadata=self.approval_metadata,
            approval_policy=self.approval_policy,
            run_store=self.run_store,
            run_journal=self.run_journal,
            trace=self.trace,
            tool_scheduler_factory=self.tool_scheduler_factory,
        )

    async def run(
        self,
        messages: Sequence[Message],
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AgentResult:
        """Run the configured scenario and return the terminal result."""

        return await self.agent().run(
            messages,
            context=context,
            stream=stream,
            controller=controller,
        )

    async def events(
        self,
        messages: Sequence[Message],
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> list[AgentEvent]:
        """Run the configured scenario and collect its event stream."""

        return await collect_events(
            self.agent(),
            messages,
            context=context,
            stream=stream,
            controller=controller,
        )
