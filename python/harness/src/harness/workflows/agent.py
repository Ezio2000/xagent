"""High-level open agent workflow facade built on kernel ports."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, TypeAlias, cast

from diagnostics import replay_trace
from kernel import (
    AgentEvent,
    AgentLoop,
    AgentResult,
    ApprovalPolicy,
    LoopLimits,
    ModelClient,
    ModelOptions,
    PauseSelector,
    ResponseFormat,
    ResumeInput,
    RunController,
    RunJournal,
    RunSnapshot,
    RunStore,
    RuntimeContext,
    RuntimeHook,
    ToolChoice,
    ToolRegistryProtocol,
    ToolSchedulerFactory,
)
from toolkit import Tool, ToolRegistry

from harness.workflows.inputs import (
    HarnessInput,
    OptionalHarnessInput,
    normalize_messages,
    normalize_optional_messages,
)
from harness.workflows.results import PausedHarnessRun, TraceHarnessRun
from harness.workflows.waiting import WaitingState, waiting_from_result

ToolSource: TypeAlias = ToolRegistryProtocol | Sequence[Tool]
PausedSource: TypeAlias = PausedHarnessRun | AgentResult | RunSnapshot


class AgentHarness:
    """Convenience workflow facade for running an agent through public SDK ports."""

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolSource | None = None,
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
        self.tools = self._normalize_tools(tools)
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

    @staticmethod
    def _normalize_tools(tools: ToolSource | None) -> ToolRegistryProtocol | None:
        if tools is None:
            return None
        if isinstance(cast(object, tools), ToolRegistryProtocol):
            return cast(ToolRegistryProtocol, tools)
        if isinstance(tools, Sequence):
            return ToolRegistry(list(tools))
        raise TypeError("tools must be a ToolRegistryProtocol or a sequence of toolkit Tool")

    def agent(self) -> AgentLoop:
        """Create a fresh `AgentLoop` for one workflow operation."""

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
        input: HarnessInput,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AgentResult:
        """Run an agent from common host input shapes."""

        return await self.agent().run(
            normalize_messages(input),
            context=context,
            stream=stream,
            controller=controller,
        )

    async def events(
        self,
        input: HarnessInput,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> list[AgentEvent]:
        """Run an agent and collect its event stream."""

        return [
            event
            async for event in self.stream_events(
                input,
                context=context,
                stream=stream,
                controller=controller,
            )
        ]

    async def stream_events(
        self,
        input: HarnessInput,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield runtime events for one agent run."""

        async for event in self.agent().run_events(
            normalize_messages(input),
            context=context,
            stream=stream,
            controller=controller,
        ):
            yield event

    async def run_until_pause(
        self,
        input: HarnessInput,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> PausedHarnessRun:
        """Run and require the result to stop at a resumable pause."""

        result = await self.run(input, context=context, stream=stream, controller=controller)
        return PausedHarnessRun.from_result(result)

    async def resume(
        self,
        paused: PausedSource,
        input: OptionalHarnessInput = None,
        *,
        expected_pause: PauseSelector | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AgentResult:
        """Resume a paused run from a paused wrapper, result, or snapshot."""

        return await self.agent().run_snapshot(
            self.resume_input(
                paused,
                input,
                expected_pause=expected_pause,
                metadata=metadata,
            ),
            stream=stream,
            controller=controller,
        )

    async def resume_events(
        self,
        paused: PausedSource,
        input: OptionalHarnessInput = None,
        *,
        expected_pause: PauseSelector | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> list[AgentEvent]:
        """Resume a paused run and collect its event stream."""

        resume_input = self.resume_input(
            paused,
            input,
            expected_pause=expected_pause,
            metadata=metadata,
        )
        return [
            event
            async for event in self.agent().run_snapshot_events(
                resume_input,
                stream=stream,
                controller=controller,
            )
        ]

    def resume_input(
        self,
        paused: PausedSource,
        input: OptionalHarnessInput = None,
        *,
        expected_pause: PauseSelector | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResumeInput:
        """Build typed resume input from common paused-run values."""

        return ResumeInput(
            snapshot=self.snapshot_from(paused),
            append_messages=normalize_optional_messages(input),
            expected_pause=expected_pause,
            metadata={} if metadata is None else metadata,
        )

    def snapshot_from(self, paused: object) -> RunSnapshot:
        """Extract a defensive-copy snapshot from a paused workflow value."""

        if isinstance(paused, PausedHarnessRun):
            return RunSnapshot.from_dict(paused.snapshot.to_dict())
        if isinstance(paused, AgentResult):
            if paused.snapshot is None:
                raise ValueError("agent result is missing a snapshot")
            return RunSnapshot.from_dict(paused.snapshot.to_dict())
        if isinstance(paused, RunSnapshot):
            return RunSnapshot.from_dict(paused.to_dict())
        raise TypeError("paused value must be PausedHarnessRun, AgentResult, or RunSnapshot")

    async def run_with_trace(
        self,
        input: HarnessInput,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
        strict: bool = True,
    ) -> TraceHarnessRun:
        """Run and validate the emitted diagnostics trace."""

        result = await self.run(input, context=context, stream=stream, controller=controller)
        if result.trace is None:
            raise ValueError("trace is disabled for this harness")
        trace = result.trace
        return TraceHarnessRun(
            result=result,
            trace=trace,
            replay=replay_trace(trace, strict=strict),
        )

    @staticmethod
    def waiting_state(
        result: AgentResult,
        *,
        events: Sequence[AgentEvent] = (),
    ) -> WaitingState:
        """Extract pause/background task information from a run."""

        return waiting_from_result(result, events=events)
