from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import cast

from jharness.kernel import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    Checkpoint,
    ContentPart,
    DeltaSink,
    Event,
    EventKind,
    Invocation,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    ToolCall,
    ToolFailure,
    ToolSuccess,
    thaw_json_value,
)
from jharness.toolkit import ToolRegistry
from jharness.tools import EditTool, ReadTool, WriteTool

ResponseFactory = Callable[[int, ModelRequest], ModelResponse]


class _DeterministicModel(Model):
    def __init__(self, respond: ResponseFactory) -> None:
        self._respond = respond
        self.turns = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        response = self._respond(self.turns, request)
        self.turns += 1
        return response


class _AllowAll(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalAllow(request.call.id) for request in requests)


class _DenyAll(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalDeny(request.call.id, "test denial") for request in requests)


def _final() -> ModelResponse:
    return ModelResponse((ContentPart.text_part("done"),), finish_reason="stop")


def _last_success(request: ModelRequest) -> dict[str, object]:
    outcome = request.messages[-1].outcome
    assert isinstance(outcome, ToolSuccess)
    structured = thaw_json_value(outcome.structured_content)
    assert isinstance(structured, dict)
    return cast(dict[str, object], structured)


def _last_visible_read_sha256(request: ModelRequest) -> str:
    outcome = request.messages[-1].outcome
    assert isinstance(outcome, ToolSuccess)
    assert len(outcome.parts) == 1
    text = outcome.parts[0].text
    assert text is not None
    prefix = "SHA-256 (raw file bytes): "
    first_line = text.partition("\n")[0]
    assert first_line.startswith(prefix)
    digest = first_line.removeprefix(prefix)
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    return digest


async def _collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


def test_runtime_read_edit_read_hands_off_model_visible_sha256(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    original = b"alpha\nbeta\n"
    updated = b"alpha\ngamma\n"
    source.write_bytes(original)
    observed: dict[str, str] = {}

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        assert {spec.name for spec in request.tools} == {"Edit", "Read"}
        if turn == 0:
            return ModelResponse(
                tool_calls=(ToolCall("read-before", "Read", {"file_path": "note.txt"}),)
            )
        if turn == 1:
            observed["read"] = _last_visible_read_sha256(request)
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "edit",
                        "Edit",
                        {
                            "file_path": "note.txt",
                            "old_string": "beta",
                            "new_string": "gamma",
                            "expected_sha256": observed["read"],
                        },
                    ),
                )
            )
        if turn == 2:
            result = _last_success(request)
            observed["edit"] = cast(str, result["sha256"])
            return ModelResponse(
                tool_calls=(ToolCall("read-after", "Read", {"file_path": "note.txt"}),)
            )
        if turn == 3:
            result = _last_success(request)
            observed["final_read"] = _last_visible_read_sha256(request)
            observed["content"] = cast(str, result["content"])
            return _final()
        raise AssertionError("unexpected model turn")

    model = _DeterministicModel(respond)
    registry = ToolRegistry((ReadTool(tmp_path), EditTool(tmp_path)))
    checkpoint, _ = asyncio.run(
        _collect(
            Runtime(model=model, tools=registry, approval_policy=_AllowAll()).start(
                (Message.user("update note.txt"),)
            )
        )
    )

    assert source.read_bytes() == updated
    assert observed == {
        "read": sha256(original).hexdigest(),
        "edit": sha256(updated).hexdigest(),
        "final_read": sha256(updated).hexdigest(),
        "content": "alpha\ngamma",
    }
    assert model.turns == 4
    assert [message.role for message in checkpoint.snapshot.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_runtime_write_then_read_observes_created_content(tmp_path: Path) -> None:
    content = "created by runtime\n"
    observed: dict[str, object] = {}

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        assert {spec.name for spec in request.tools} == {"Read", "Write"}
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "write",
                        "Write",
                        {
                            "file_path": "created.txt",
                            "content": content,
                            "expected_sha256": None,
                        },
                    ),
                )
            )
        result = _last_success(request)
        if turn == 1:
            observed["write_sha256"] = result["sha256"]
            observed["operation"] = result["operation"]
            return ModelResponse(
                tool_calls=(ToolCall("read", "Read", {"file_path": "created.txt"}),)
            )
        if turn == 2:
            observed["read_sha256"] = result["sha256"]
            observed["content"] = result["content"]
            return _final()
        raise AssertionError("unexpected model turn")

    registry = ToolRegistry((WriteTool(tmp_path), ReadTool(tmp_path)))
    checkpoint, _ = asyncio.run(
        _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=registry,
                approval_policy=_AllowAll(),
            ).start((Message.user("create and read a file"),))
        )
    )

    expected_digest = sha256(content.encode()).hexdigest()
    assert (tmp_path / "created.txt").read_bytes() == content.encode()
    assert observed == {
        "write_sha256": expected_digest,
        "operation": "created",
        "read_sha256": expected_digest,
        "content": "created by runtime",
    }
    assert all(
        isinstance(message.outcome, ToolSuccess)
        for message in checkpoint.snapshot.history
        if message.role == "tool"
    )


def test_runtime_approval_deny_does_not_write_to_disk(tmp_path: Path) -> None:
    target = tmp_path / "denied.txt"

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            assert [spec.name for spec in request.tools] == ["Write"]
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "denied-write",
                        "Write",
                        {
                            "file_path": "denied.txt",
                            "content": "must not exist",
                            "expected_sha256": None,
                        },
                    ),
                )
            )
        if turn == 1:
            outcome = request.messages[-1].outcome
            assert isinstance(outcome, ToolFailure)
            assert outcome.error.code == "approval_denied"
            return _final()
        raise AssertionError("unexpected model turn")

    checkpoint, events = asyncio.run(
        _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=ToolRegistry((WriteTool(tmp_path),)),
                approval_policy=_DenyAll(),
            ).start((Message.user("attempt a denied write"),))
        )
    )

    assert not target.exists()
    outcome = checkpoint.snapshot.history[2].outcome
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == "approval_denied"
    assert EventKind.APPROVAL_REQUESTED in [event.kind for event in events]
    assert EventKind.APPROVAL_DECIDED in [event.kind for event in events]
    assert EventKind.TOOL_STARTED not in [event.kind for event in events]


def test_runtime_two_writes_are_serial_and_report_nonparallel_starts(tmp_path: Path) -> None:
    first = b"first\n"
    second = "second\n"

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "write-first",
                        "Write",
                        {
                            "file_path": "serial.txt",
                            "content": first.decode(),
                            "expected_sha256": None,
                        },
                    ),
                    ToolCall(
                        "write-second",
                        "Write",
                        {
                            "file_path": "serial.txt",
                            "content": second,
                            "expected_sha256": sha256(first).hexdigest(),
                        },
                    ),
                )
            )
        if turn == 1:
            outcomes = [message.outcome for message in request.messages if message.role == "tool"]
            assert len(outcomes) == 2
            assert all(isinstance(outcome, ToolSuccess) for outcome in outcomes)
            return _final()
        raise AssertionError("unexpected model turn")

    checkpoint, events = asyncio.run(
        _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=ToolRegistry((WriteTool(tmp_path),)),
                approval_policy=_AllowAll(),
            ).start((Message.user("perform two writes"),))
        )
    )

    lifecycle: list[tuple[str, str]] = []
    starts = [event for event in events if event.kind is EventKind.TOOL_STARTED]
    for event in events:
        if event.kind is EventKind.TOOL_STARTED:
            call = event.data["call"]
            assert isinstance(call, Mapping)
            lifecycle.append(("started", cast(str, call["id"])))
        elif event.kind is EventKind.TOOL_FINISHED:
            lifecycle.append(("finished", cast(str, event.data["tool_call_id"])))

    assert (tmp_path / "serial.txt").read_text() == second
    assert [cast(bool, event.data["parallel"]) for event in starts] == [False, False]
    assert lifecycle == [
        ("started", "write-first"),
        ("finished", "write-first"),
        ("started", "write-second"),
        ("finished", "write-second"),
    ]
    assert all(
        isinstance(message.outcome, ToolSuccess)
        for message in checkpoint.snapshot.history
        if message.role == "tool"
    )
