"""Persistent conversation history and reduction ports."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, overload, runtime_checkable

from jharness.kernel._digest import append_history_digest, empty_history_digest
from jharness.kernel._validation import (
    expect_instance,
    expect_instance_tuple,
    expect_int,
    expect_non_empty_str,
    freeze_mapping,
)
from jharness.kernel.messages import Message
from jharness.kernel.state import RunState

if TYPE_CHECKING:
    from jharness.kernel.snapshot import RunSnapshot


@dataclass(frozen=True, slots=True)
class _HistoryLeaf:
    message: Message


@dataclass(frozen=True, slots=True)
class _HistoryNode:
    message: Message
    left: _HistoryLeaf | _HistoryNode
    right: _HistoryLeaf | _HistoryNode


_HistoryTree: TypeAlias = _HistoryLeaf | _HistoryNode


@dataclass(frozen=True, slots=True)
class _HistoryDigit:
    weight: int
    tree: _HistoryTree
    next: _HistoryDigit | None


@dataclass(frozen=True, slots=True, init=False, repr=False)
class RunHistory(Sequence[Message]):
    """Immutable skew-binary history with constant-time append operations."""

    _digits: _HistoryDigit
    _count: int
    _digest: bytes
    _first: Message

    def __init__(self, messages: Sequence[Message]) -> None:
        normalized = expect_instance_tuple(messages, Message, "run history")
        if not normalized:
            raise ValueError("run history must not be empty")
        digits: _HistoryDigit | None = None
        digest = empty_history_digest()
        for message in normalized:
            digest = append_history_digest(digest, message)
            digits = _cons(digits, message)
        assert digits is not None
        object.__setattr__(self, "_digits", digits)
        object.__setattr__(self, "_count", len(normalized))
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_first", normalized[0])

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Message]:
        return self._iter_window(0, self._count)

    def __reversed__(self) -> Iterator[Message]:
        digit: _HistoryDigit | None = self._digits
        while digit is not None:
            yield from _iter_tree(digit.tree)
            digit = digit.next

    @property
    def first(self) -> Message:
        """Return the first message in constant time."""

        return self._first

    @property
    def digest(self) -> bytes:
        """Return the canonical incremental history digest."""

        return self._digest

    @overload
    def __getitem__(self, index: int) -> Message: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Message, ...]: ...

    def __getitem__(self, index: int | slice) -> Message | tuple[Message, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step == 1:
                return tuple(self._iter_window(start, stop))
            return tuple(self)[index]
        index = expect_int(index, "history index")
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("history index out of range")
        if index == 0:
            return self._first
        return self._lookup_reverse(self._count - index - 1)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, RunHistory):
            return False
        if self._count != other._count or self._digest != other._digest:
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"RunHistory({tuple(self)!r})"

    def count(self, value: Message) -> int:
        """Count equal messages with one chronological scan."""

        return sum(message == value for message in self)

    def index(self, value: Message, start: int = 0, stop: int | None = None) -> int:
        """Find a message with one bounded chronological scan."""

        start = expect_int(start, "history index start")
        stop = self._count if stop is None else expect_int(stop, "history index stop")
        normalized_start, normalized_stop, _ = slice(start, stop).indices(self._count)
        for index, message in enumerate(
            self._iter_window(normalized_start, normalized_stop),
            start=normalized_start,
        ):
            if message == value:
                return index
        raise ValueError(f"{value!r} is not in RunHistory")

    def iter_tail(self, count: int) -> Iterator[Message]:
        """Iterate at most ``count`` trailing messages without walking the prefix."""

        count = expect_int(count, "history tail count")
        if count < 0:
            raise ValueError("history tail count must be >= 0")
        newest_first = tuple(islice(reversed(self), min(count, self._count)))
        return reversed(newest_first)

    def iter_window(self, start: int, stop: int | None = None) -> Iterator[Message]:
        """Iterate one normalized half-open history window in chronological order."""

        start = expect_int(start, "history window start")
        stop = self._count if stop is None else expect_int(stop, "history window stop")
        if start < 0 or stop < start or stop > self._count:
            raise ValueError("history window must satisfy 0 <= start <= stop <= len(history)")
        return self._iter_window(start, stop)

    def _iter_window(self, start: int, stop: int) -> Iterator[Message]:
        if start == stop:
            return iter(())
        reverse_start = self._count - stop
        reverse_stop = self._count - start
        selected: list[tuple[_HistoryTree, int, int, int]] = []
        digit: _HistoryDigit | None = self._digits
        offset = 0
        while digit is not None and offset < reverse_stop:
            digit_stop = offset + digit.weight
            local_start = max(reverse_start, offset) - offset
            local_stop = min(reverse_stop, digit_stop) - offset
            if local_start < local_stop:
                selected.append((digit.tree, digit.weight, local_start, local_stop))
            offset = digit_stop
            digit = digit.next

        def messages() -> Iterator[Message]:
            for tree, weight, local_start, local_stop in reversed(selected):
                yield from _iter_tree_reverse_window(tree, weight, local_start, local_stop)

        return messages()

    def _lookup_reverse(self, index: int) -> Message:
        digit: _HistoryDigit | None = self._digits
        while digit is not None:
            if index < digit.weight:
                return _lookup_tree(digit.tree, digit.weight, index)
            index -= digit.weight
            digit = digit.next
        raise AssertionError("history random-access structure is inconsistent")

    def _append(self, messages: tuple[Message, ...], *, digest: bytes) -> RunHistory:
        if not messages:
            return self
        digits = self._digits
        for message in messages:
            digits = _cons(digits, message)
        history = object.__new__(RunHistory)
        object.__setattr__(history, "_digits", digits)
        object.__setattr__(history, "_count", self._count + len(messages))
        object.__setattr__(history, "_digest", digest)
        object.__setattr__(history, "_first", self._first)
        return history

    def _digest_bytes(self) -> bytes:
        return self._digest


def _cons(digits: _HistoryDigit | None, message: Message) -> _HistoryDigit:
    if digits is not None and digits.next is not None and digits.weight == digits.next.weight:
        second = digits.next
        return _HistoryDigit(
            digits.weight * 2 + 1,
            _HistoryNode(message, digits.tree, second.tree),
            second.next,
        )
    return _HistoryDigit(1, _HistoryLeaf(message), digits)


def _lookup_tree(tree: _HistoryTree, weight: int, index: int) -> Message:
    while isinstance(tree, _HistoryNode):
        if index == 0:
            return tree.message
        child_weight = (weight - 1) // 2
        if index <= child_weight:
            tree = tree.left
            index -= 1
        else:
            tree = tree.right
            index -= child_weight + 1
        weight = child_weight
    if index != 0:
        raise AssertionError("history tree weight is inconsistent")
    return tree.message


def _iter_tree(tree: _HistoryTree) -> Iterator[Message]:
    yield tree.message
    if isinstance(tree, _HistoryNode):
        yield from _iter_tree(tree.left)
        yield from _iter_tree(tree.right)


def _iter_tree_reverse_window(
    tree: _HistoryTree,
    weight: int,
    start: int,
    stop: int,
) -> Iterator[Message]:
    def visit(item: _HistoryTree, item_weight: int, offset: int) -> Iterator[Message]:
        if stop <= offset or offset + item_weight <= start:
            return
        if isinstance(item, _HistoryLeaf):
            yield item.message
            return
        child_weight = (item_weight - 1) // 2
        yield from visit(item.right, child_weight, offset + child_weight + 1)
        yield from visit(item.left, child_weight, offset + 1)
        if start <= offset < stop:
            yield item.message

    return visit(tree, weight, 0)


@dataclass(frozen=True, slots=True)
class HistoryRewrite:
    """A reducer proposal that may replace history at a plan boundary."""

    messages: RunHistory
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_instance(self.messages, RunHistory, "history rewrite messages")
        expect_non_empty_str(self.reason, "history rewrite reason")
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "history rewrite metadata"),
        )


@runtime_checkable
class HistoryReducer(Protocol):
    """Propose a valid history with no more messages at a planning boundary."""

    async def reduce(self, snapshot: RunSnapshot) -> HistoryRewrite | None: ...


def validate_history(history: Sequence[Message], state: RunState) -> None:
    """Fully validate ordered tool linkage against one lifecycle state."""

    from jharness.kernel._history import analyze_history

    analyze_history(history, state)
