from __future__ import annotations

import pytest
from harness import (
    timeline_event_label,
)
from kernel import (
    AgentEvent,
    AgentLoop,
    ApprovalDecision,
    ApprovalRequest,
    ContentPart,
    EventTypes,
    JournalRecord,
    Message,
    ModelResponse,
    RuntimeContext,
    ToolCall,
    ToolSpec,
)
from support import (
    ApprovalPolicyByCall,
    FailingApprovalPolicy,
    FailingCheckpointJournal,
    FailingRunStore,
    FailingSecondCheckpointStore,
    MemoryRunJournal,
    MemoryRunStore,
    ScriptedModel,
    SequencedApprovalPolicy,
    StaticApprovalPolicy,
    TimelineRunJournal,
)


@pytest.mark.asyncio
async def test_memory_run_store_records_checkpoint_copies() -> None:
    store = MemoryRunStore()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=store,
    ).run([Message.user([ContentPart.text_part("hello")])])

    assert result.snapshot is not None
    assert len(store.checkpoints) == 1
    loaded = await store.load_checkpoint(result.run_id)

    assert loaded.to_dict() == result.snapshot.to_dict()
    assert await store.list_checkpoints(result.run_id) == [store.checkpoints[0].summary()]


@pytest.mark.asyncio
async def test_failing_second_checkpoint_store_keeps_first_checkpoint() -> None:
    memory_store = MemoryRunStore()
    await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=memory_store,
    ).run([Message.user([ContentPart.text_part("hello")])])

    store = FailingSecondCheckpointStore()
    await store.save_checkpoint(memory_store.checkpoints[0])
    assert len(store.checkpoints) == 1
    with pytest.raises(RuntimeError, match="store unavailable"):
        await store.save_checkpoint(store.checkpoints[0])


@pytest.mark.asyncio
async def test_memory_run_journal_records_and_reads_record_copies() -> None:
    journal = MemoryRunJournal()
    event = AgentEvent(EventTypes.RUN_STARTED, {"state": {}}, run_id="run-1", sequence=1)
    await journal.append(JournalRecord(event=event, checkpoint_id=None))

    records = [record async for record in journal.read("run-1")]

    assert len(records) == 1
    assert records[0].event_type == EventTypes.RUN_STARTED
    assert records[0] is not journal.records[0]


@pytest.mark.asyncio
async def test_timeline_and_failing_journal_variants() -> None:
    timeline: list[str] = []
    journal = TimelineRunJournal(timeline)
    event = AgentEvent(
        EventTypes.STATE_CHANGED,
        {"from": "planning", "to": "completed"},
        run_id="run-1",
        sequence=1,
    )

    assert timeline_event_label("caller", event) == "caller:state_changed:completed"
    await journal.append(JournalRecord(event=event, checkpoint_id=None))
    assert timeline == ["journal:state_changed:completed"]

    failing = FailingCheckpointJournal()
    checkpoint = AgentEvent(EventTypes.CHECKPOINT, {"state": {}}, run_id="run-1", sequence=2)
    with pytest.raises(RuntimeError, match="journal unavailable"):
        await failing.append(JournalRecord(event=checkpoint, checkpoint_id="checkpoint-2"))


@pytest.mark.asyncio
async def test_failure_port_variants_raise_expected_errors() -> None:
    memory_store = MemoryRunStore()
    await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=memory_store,
    ).run([Message.user([ContentPart.text_part("hello")])])

    with pytest.raises(RuntimeError, match="store unavailable"):
        await FailingRunStore().save_checkpoint(memory_store.checkpoints[0])

    policy = FailingApprovalPolicy()
    with pytest.raises(RuntimeError, match="approval backend unavailable"):
        await policy.decide(_approval_request("call-1"))


@pytest.mark.asyncio
async def test_scripted_approval_policies_record_requests() -> None:
    static = StaticApprovalPolicy(ApprovalDecision.deny("blocked"))
    by_call = ApprovalPolicyByCall({"call-2": ApprovalDecision.pause("approval_required")})
    sequenced = SequencedApprovalPolicy([ApprovalDecision.deny("first")])

    assert (await static.decide(_approval_request("call-1"))).action == "deny"
    assert (await by_call.decide(_approval_request("call-1"))).action == "allow"
    assert (await by_call.decide(_approval_request("call-2"))).action == "pause"
    assert (await sequenced.decide(_approval_request("call-1"))).reason == "first"
    assert (await sequenced.decide(_approval_request("call-1"))).action == "allow"

    assert [request.tool_call.id for request in static.requests] == ["call-1"]
    assert [request.tool_call.id for request in by_call.requests] == ["call-1", "call-2"]
    assert [request.tool_call.id for request in sequenced.requests] == ["call-1", "call-1"]


def _approval_request(call_id: str) -> ApprovalRequest:
    return ApprovalRequest(
        tool_call=ToolCall(id=call_id, name="record", arguments={}),
        tool_spec=ToolSpec(
            name="record",
            description="Record.",
            input_schema={"type": "object", "properties": {}},
        ),
        context=RuntimeContext(run_id="run-1", started_at=1.0),
        risk={},
        metadata={},
    )
