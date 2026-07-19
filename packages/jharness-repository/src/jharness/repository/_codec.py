"""Canonical checkpoint encoding shared by repository adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from jharness.kernel import Checkpoint, ProtocolError, RepositoryError
from jharness.kernel.wire import decode_checkpoint as decode_wire_checkpoint
from jharness.kernel.wire import encode_checkpoint as encode_wire_checkpoint


@dataclass(frozen=True, slots=True)
class EncodedCheckpoint:
    """The indexed fields and canonical bytes for one checkpoint."""

    checkpoint_id: str
    run_id: str
    revision: int
    payload: bytes
    digest: bytes

    @property
    def expected_revision(self) -> int | None:
        """Return the head revision required immediately before this checkpoint."""
        return self.revision - 1 if self.revision else None


def encode_checkpoint(checkpoint: Checkpoint) -> EncodedCheckpoint:
    """Encode a checkpoint into stable JSON bytes and a content fingerprint."""
    if not isinstance(cast(object, checkpoint), Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    checkpoint_id = str.__str__(checkpoint.id)
    run_id = str.__str__(checkpoint.snapshot.context.run_id)
    if not checkpoint_id:
        raise ValueError("checkpoint id must not be empty")
    if not run_id:
        raise ValueError("run id must not be empty")
    payload = json.dumps(
        encode_wire_checkpoint(checkpoint),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return EncodedCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        revision=checkpoint.snapshot.revision,
        payload=payload,
        digest=sha256(payload).digest(),
    )


def decode_checkpoint(payload: bytes) -> Checkpoint:
    """Decode stored checkpoint bytes, reporting malformed storage uniformly."""
    if type(payload) is not bytes:
        raise TypeError("checkpoint payload must be bytes")
    try:
        document: object = json.loads(payload)
        return decode_wire_checkpoint(document)
    except (
        OverflowError,
        RecursionError,
        UnicodeDecodeError,
        ProtocolError,
        TypeError,
        ValueError,
    ) as exc:
        raise RepositoryError("stored checkpoint payload is invalid") from exc
