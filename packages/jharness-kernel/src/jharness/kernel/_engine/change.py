"""Closed durable change consumed by the pure reducer."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, cast

from jharness.kernel._validation import expect_instance, expect_instance_tuple
from jharness.kernel.checkpoint import (
    ControlFact,
    ConversationInsertFact,
    Fact,
    FailedControl,
    LimitedControl,
    SuspendedControl,
)
from jharness.kernel.control import Insert
from jharness.kernel.limits import LimitReason
from jharness.kernel.messages import ErrorInfo, Message
from jharness.kernel.models import ModelUsage
from jharness.kernel.state import (
    ActiveState,
    Failed,
    Limited,
    Planning,
    RunState,
    Suspended,
    Suspension,
)

if TYPE_CHECKING:
    from jharness.kernel.checkpoint import Checkpoint
    from jharness.kernel.snapshot import RunSnapshot


@dataclass(frozen=True, slots=True)
class Change:
    fact: Fact
    state: RunState
    append: tuple[Message, ...] = ()
    replace: tuple[Message, ...] | None = None
    planning_steps: int = 0
    tool_calls: int = 0
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.fact), Fact):
            raise TypeError("change fact must be a Fact")
        if not isinstance(cast(object, self.state), RunState):
            raise TypeError("change state must be a RunState")
        append = expect_instance_tuple(self.append, Message, "change append")
        replace = (
            None
            if self.replace is None
            else expect_instance_tuple(self.replace, Message, "change replacement")
        )
        if replace is not None and (not replace or append):
            raise ValueError("change replacement must be non-empty and exclusive with append")
        object.__setattr__(self, "append", append)
        object.__setattr__(self, "replace", replace)


def failed(code: str, message: str) -> Change:
    return Change(
        ControlFact(time(), FailedControl(code)),
        Failed(ErrorInfo(code, message)),
    )


def limited(reason: LimitReason) -> Change:
    return Change(ControlFact(time(), LimitedControl(reason)), Limited(reason))


def insert(control: Insert) -> Change:
    return Change(
        ConversationInsertFact(time(), control.source),
        Planning(),
        append=(control.message,),
    )


def suspend(state: ActiveState, suspension: Suspension) -> Change:
    return Change(
        ControlFact(
            time(),
            SuspendedControl(
                suspension.reason,
                suspension.source,
                suspension.wait_id,
                tuple(sorted(suspension.metadata)),
            ),
        ),
        Suspended(state, suspension),
    )


def reduce(snapshot: RunSnapshot, change: Change, *, checkpoint_id: str) -> Checkpoint:
    """Purely apply one trusted change and return the next Checkpoint."""

    from jharness.kernel.checkpoint import Checkpoint
    from jharness.kernel.snapshot import RunSnapshot

    snapshot = expect_instance(snapshot, RunSnapshot, "snapshot")
    change = expect_instance(change, Change, "change")
    metrics = snapshot.metrics.advance(
        planning_steps=change.planning_steps,
        tool_calls=change.tool_calls,
        usage=change.usage,
    )
    next_snapshot = snapshot._evolve(  # pyright: ignore[reportPrivateUsage]
        append=change.append,
        replace=change.replace,
        metrics=metrics,
        state=change.state,
    )
    return Checkpoint(checkpoint_id, next_snapshot, change.fact)
