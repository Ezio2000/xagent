"""Public immutable Runtime assembly and direct invocation entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from time import time
from typing import Any, cast
from uuid import uuid4

from jharness.kernel._engine.engine import Engine, EngineConfig
from jharness.kernel._validation import (
    expect_instance,
    expect_number,
)
from jharness.kernel.approval import ApprovalPolicy
from jharness.kernel.checkpoint import Checkpoint
from jharness.kernel.commands import (
    ContinueRequest,
    ResumeRequest,
    RunRequest,
    StartRequest,
    SuspensionSelector,
)
from jharness.kernel.context import RunContext
from jharness.kernel.history import HistoryReducer
from jharness.kernel.invocation import Invocation
from jharness.kernel.limits import RunLimits
from jharness.kernel.messages import Message
from jharness.kernel.models import Model, ModelOptions, ResponseFormat, ToolChoice
from jharness.kernel.repository import RunRepository
from jharness.kernel.tools import (
    BatchPolicy,
    DefaultBatchPolicy,
    EmptyToolCatalogProvider,
    ToolCatalogProvider,
)


class Runtime:
    """Immutable configuration that creates single-use Invocations."""

    __slots__ = ("_config",)

    def __init__(
        self,
        *,
        model: Model,
        tools: ToolCatalogProvider | None = None,
        limits: RunLimits | None = None,
        model_options: ModelOptions | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        approval_policy: ApprovalPolicy | None = None,
        history_reducer: HistoryReducer | None = None,
        batch_policy: BatchPolicy | None = None,
        repository: RunRepository | None = None,
        repository_timeout: float = 5.0,
    ) -> None:
        model = expect_instance(model, Model, "model")
        tools = (
            EmptyToolCatalogProvider()
            if tools is None
            else expect_instance(tools, ToolCatalogProvider, "tools")
        )
        limits = RunLimits() if limits is None else expect_instance(limits, RunLimits, "limits")
        options = (
            ModelOptions()
            if model_options is None
            else expect_instance(model_options, ModelOptions, "model_options")
        )
        choice = (
            ToolChoice()
            if tool_choice is None
            else expect_instance(tool_choice, ToolChoice, "tool_choice")
        )
        if response_format is not None:
            expect_instance(response_format, ResponseFormat, "response_format")
        if approval_policy is not None:
            expect_instance(approval_policy, ApprovalPolicy, "approval_policy")
        if history_reducer is not None:
            expect_instance(history_reducer, HistoryReducer, "history_reducer")
        policy = (
            DefaultBatchPolicy()
            if batch_policy is None
            else expect_instance(batch_policy, BatchPolicy, "batch_policy")
        )
        if repository is not None:
            expect_instance(repository, RunRepository, "repository")
        repository_timeout = expect_number(repository_timeout, "repository_timeout")
        if repository_timeout <= 0:
            raise ValueError("repository_timeout must be > 0")
        self._config = EngineConfig(
            model=model,
            tools=tools,
            limits=limits,
            model_options=options,
            tool_choice=choice,
            response_format=response_format,
            approval=approval_policy,
            history_reducer=history_reducer,
            batch_policy=policy,
            repository=repository,
            repository_timeout=repository_timeout,
        )

    def start(
        self,
        messages: Sequence[Message],
        *,
        context: RunContext | None = None,
        stream: bool = False,
    ) -> Invocation:
        context = self._start_context(context)
        request = StartRequest(tuple(messages), context)
        return self._invocation(request, context.run_id, stream)

    def continue_from(self, checkpoint: Checkpoint, *, stream: bool = False) -> Invocation:
        request = ContinueRequest(checkpoint)
        return self._invocation(
            request,
            request.checkpoint.snapshot.context.run_id,
            stream,
        )

    def resume(
        self,
        checkpoint: Checkpoint,
        *,
        selector: SuspensionSelector | None = None,
        append_messages: Sequence[Message] = (),
        metadata: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> Invocation:
        request = ResumeRequest(
            checkpoint,
            selector,
            tuple(append_messages),
            {} if metadata is None else metadata,
        )
        return self._invocation(
            request,
            request.checkpoint.snapshot.context.run_id,
            stream,
        )

    def _invocation(self, request: RunRequest, run_id: str, stream: bool) -> Invocation:
        if not isinstance(cast(object, stream), bool):
            raise TypeError("stream must be bool")
        engine = Engine(self._config, request, stream=stream)
        return Invocation(run_id, engine.run, stream=stream)

    def _start_context(self, context: RunContext | None) -> RunContext:
        now = time()
        if context is None:
            return RunContext(
                str(uuid4()),
                now,
                None
                if self._config.limits.timeout_seconds is None
                else now + self._config.limits.timeout_seconds,
            )
        context = expect_instance(context, RunContext, "context")
        deadline = context.deadline
        if self._config.limits.timeout_seconds is not None:
            configured = now + self._config.limits.timeout_seconds
            deadline = configured if deadline is None else min(deadline, configured)
        return replace(context, deadline=deadline)
