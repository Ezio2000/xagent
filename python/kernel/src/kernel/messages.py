"""Message and content protocol types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from kernel._validation import (
    expect_mapping as _expect_mapping,
)
from kernel._validation import (
    expect_optional_int as _expect_optional_int,
)
from kernel._validation import (
    expect_optional_str as _expect_optional_str,
)
from kernel._validation import (
    expect_present_str as _expect_present_str,
)
from kernel._validation import (
    expect_sequence as _expect_sequence,
)
from kernel._validation import (
    expect_str as _expect_str,
)
from kernel._validation import (
    reject_unknown_keys as _reject_unknown_keys,
)

Role = Literal["system", "user", "assistant", "tool", "external"]
KNOWN_ROLES = {"system", "user", "assistant", "tool", "external"}
ToolCallMode = str
PartType = str


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _empty_parts() -> list[ContentPart]:
    return []


def _empty_tool_calls() -> list[ToolCall]:
    return []


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


def _expect_tool_call_mode(value: object, label: str) -> ToolCallMode:
    mode = _expect_str(value, label)
    if not mode:
        raise ValueError(f"{label} must not be empty")
    return mode


@dataclass(slots=True, frozen=True)
class ArtifactRef:
    """Host-owned artifact reference carried by message content parts.

    The runtime stores only the reference and integrity metadata. It does not
    read, write, retain, or dereference artifact payloads.
    """

    ref: str
    media_type: str | None = None
    name: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        ref = _expect_str(self.ref, "artifact ref")
        if not ref:
            raise ValueError("artifact ref must not be empty")
        media_type = _expect_optional_str(self.media_type, "artifact media_type")
        name = _expect_optional_str(self.name, "artifact name")
        size_bytes = _expect_optional_int(self.size_bytes, "artifact size_bytes")
        sha256 = _expect_optional_str(self.sha256, "artifact sha256")
        if media_type == "":
            raise ValueError("artifact media_type must not be empty")
        if name == "":
            raise ValueError("artifact name must not be empty")
        if size_bytes is not None and size_bytes < 0:
            raise ValueError("artifact size_bytes must be >= 0")
        if sha256 is not None and not sha256:
            raise ValueError("artifact sha256 must not be empty")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(_expect_mapping(self.metadata, "artifact metadata")),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        known = {"ref", "media_type", "name", "size_bytes", "sha256", "metadata"}
        _reject_unknown_keys(value, known, "artifact ref")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            ref=_expect_str(value["ref"], "artifact ref"),
            media_type=_expect_optional_str(value.get("media_type"), "artifact media_type"),
            name=_expect_optional_str(value.get("name"), "artifact name"),
            size_bytes=_expect_optional_int(value.get("size_bytes"), "artifact size_bytes"),
            sha256=_expect_optional_str(value.get("sha256"), "artifact sha256"),
            metadata=_expect_mapping(raw_metadata, "artifact metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"ref": self.ref}
        if self.media_type is not None:
            data["media_type"] = self.media_type
        if self.name is not None:
            data["name"] = self.name
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


@dataclass(slots=True)
class ContentPart:
    """A normalized multimodal content part.

    The runtime provides helpers for known `text`, `image`, and `file` parts,
    while the wire contract keeps `type` open for provider or SDK extensions.
    Media bytes are always referenced by URI or artifact ref, not processed by
    the core loop.
    """

    type: PartType
    text: str | None = None
    uri: str | None = None
    ref: str | None = None
    media_type: str | None = None
    name: str | None = None
    data: Mapping[str, Any] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        self.type = _expect_str(self.type, "content part type")
        self.text = _expect_optional_str(self.text, "content part text")
        self.uri = _expect_optional_str(self.uri, "content part uri")
        self.ref = _expect_optional_str(self.ref, "content part ref")
        self.media_type = _expect_optional_str(self.media_type, "content part media_type")
        self.name = _expect_optional_str(self.name, "content part name")
        if not self.type:
            raise ValueError("part type must be a non-empty string")
        if self.type == "text" and self.text is None:
            raise ValueError("text parts require text")
        if self.uri is not None and self.ref is not None:
            raise ValueError("content part cannot set both uri and ref")
        self.data = _copy_mapping(self.data)
        self.metadata = _copy_mapping(self.metadata)
        raw_artifact = self.data.get("artifact")
        if self.type == "artifact" and (self.ref is None or raw_artifact is None):
            raise ValueError("artifact parts require ref and data.artifact")
        if raw_artifact is not None:
            artifact = ArtifactRef.from_dict(_expect_mapping(raw_artifact, "content artifact"))
            if self.ref is not None and artifact.ref != self.ref:
                raise ValueError("content artifact ref must match content part ref")
            self.data["artifact"] = artifact.to_dict()

    @classmethod
    def text_part(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(type="text", text=text, metadata=metadata or {})

    @classmethod
    def image_uri(
        cls,
        uri: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="image",
            uri=uri,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
        )

    @classmethod
    def image_ref(
        cls,
        ref: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="image",
            ref=ref,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
        )

    @classmethod
    def file_uri(
        cls,
        uri: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="file",
            uri=uri,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
        )

    @classmethod
    def file_ref(
        cls,
        ref: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="file",
            ref=ref,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
        )

    @classmethod
    def artifact_ref(
        cls,
        artifact: ArtifactRef,
        *,
        part_type: str = "artifact",
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        canonical = ArtifactRef.from_dict(artifact.to_dict())
        return cls(
            type=part_type,
            ref=canonical.ref,
            media_type=canonical.media_type,
            name=canonical.name,
            data={"artifact": canonical.to_dict()},
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContentPart:
        known = {"type", "text", "uri", "ref", "media_type", "name", "data", "metadata"}
        _reject_unknown_keys(value, known, "content part")
        raw_data: object = value.get("data", {})
        raw_metadata: object = value.get("metadata", {})
        return cls(
            type=_expect_str(value["type"], "content part type"),
            text=_expect_present_str(value, "text", "content part text"),
            uri=_expect_present_str(value, "uri", "content part uri"),
            ref=_expect_present_str(value, "ref", "content part ref"),
            media_type=_expect_present_str(value, "media_type", "content part media_type"),
            name=_expect_present_str(value, "name", "content part name"),
            data=_expect_mapping(raw_data, "content part data"),
            metadata=_expect_mapping(raw_metadata, "content part metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            data["text"] = self.text
        if self.uri is not None:
            data["uri"] = self.uri
        if self.ref is not None:
            data["ref"] = self.ref
        if self.media_type is not None:
            data["media_type"] = self.media_type
        if self.name is not None:
            data["name"] = self.name
        if self.data:
            data["data"] = _copy_mapping(self.data)
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


@dataclass(slots=True)
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    mode: ToolCallMode = "execute"
    arguments: Mapping[str, Any] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        self.id = _expect_str(self.id, "tool call id")
        self.name = _expect_str(self.name, "tool call name")
        self.mode = _expect_tool_call_mode(self.mode, "tool call mode")
        if not self.id:
            raise ValueError("tool call id must not be empty")
        if not self.name:
            raise ValueError("tool call name must not be empty")
        self.arguments = _copy_mapping(self.arguments)
        self.metadata = _copy_mapping(self.metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolCall:
        known = {"id", "name", "mode", "arguments", "metadata"}
        _reject_unknown_keys(value, known, "tool call")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            id=_expect_str(value["id"], "tool call id"),
            name=_expect_str(value["name"], "tool call name"),
            mode=_expect_tool_call_mode(value["mode"], "tool call mode"),
            arguments=_expect_mapping(value["arguments"], "tool call arguments"),
            metadata=_expect_mapping(raw_metadata, "tool call metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "arguments": _copy_mapping(self.arguments),
        }
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


@dataclass(slots=True)
class Message:
    """A conversation message used by the runtime."""

    role: Role
    parts: list[ContentPart] = field(default_factory=_empty_parts)
    tool_calls: list[ToolCall] = field(default_factory=_empty_tool_calls)
    tool_call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        self.role = cast(Role, _expect_str(self.role, "message role"))
        self.tool_call_id = _expect_optional_str(self.tool_call_id, "message tool_call_id")
        if self.role not in KNOWN_ROLES:
            raise ValueError(f"unsupported message role: {self.role}")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may set tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may include tool_calls")
        self.parts = [ContentPart.from_dict(part.to_dict()) for part in self.parts]
        self.tool_calls = [ToolCall.from_dict(call.to_dict()) for call in self.tool_calls]
        tool_call_ids = [call.id for call in self.tool_calls]
        if len(tool_call_ids) != len(set(tool_call_ids)):
            raise ValueError("assistant tool_call ids must be unique")
        self.metadata = _copy_mapping(self.metadata)

    @classmethod
    def system(
        cls, parts: Sequence[ContentPart], *, metadata: Mapping[str, Any] | None = None
    ) -> Message:
        return cls(role="system", parts=list(parts), metadata=metadata or {})

    @classmethod
    def user(
        cls, parts: Sequence[ContentPart], *, metadata: Mapping[str, Any] | None = None
    ) -> Message:
        return cls(role="user", parts=list(parts), metadata=metadata or {})

    @classmethod
    def assistant(
        cls,
        parts: Sequence[ContentPart],
        tool_calls: Sequence[ToolCall] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            role="assistant",
            parts=list(parts),
            tool_calls=list(tool_calls or ()),
            metadata=metadata or {},
        )

    @classmethod
    def tool(
        cls,
        parts: Sequence[ContentPart],
        tool_call_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            role="tool",
            parts=list(parts),
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )

    @classmethod
    def external(
        cls,
        parts: Sequence[ContentPart],
        *,
        insert_id: str,
        source: str,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        external_metadata = dict(metadata or {})
        external_metadata["insert_id"] = insert_id
        external_metadata["source"] = source
        if correlation_id is not None:
            external_metadata["correlation_id"] = correlation_id
        return cls(role="external", parts=list(parts), metadata=external_metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        known = {"role", "parts", "tool_call_id", "tool_calls", "metadata"}
        _reject_unknown_keys(value, known, "message")
        role = _expect_str(value["role"], "message role")
        if role not in KNOWN_ROLES:
            raise ValueError(f"unsupported message role: {role}")
        raw_tool_calls: object = value.get("tool_calls", [])
        raw_metadata: object = value.get("metadata", {})
        return cls(
            role=cast(Role, role),
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "message part"))
                for part in _expect_sequence(value["parts"], "message parts")
            ],
            tool_calls=[
                ToolCall.from_dict(_expect_mapping(call, "message tool call"))
                for call in _expect_sequence(raw_tool_calls, "message tool_calls")
            ],
            tool_call_id=_expect_optional_str(value.get("tool_call_id"), "message tool_call_id"),
            metadata=_expect_mapping(raw_metadata, "message metadata"),
        )

    @property
    def text(self) -> str:
        return "".join(part.text or "" for part in self.parts if part.type == "text")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


def content_parts_summary(parts: Sequence[ContentPart]) -> dict[str, Any]:
    types: list[str] = []
    seen_types: set[str] = set()
    text_length = 0
    for part in parts:
        if part.type not in seen_types:
            types.append(part.type)
            seen_types.add(part.type)
        if part.type == "text" and part.text:
            text_length += len(part.text)
    return {
        "part_count": len(parts),
        "part_types": types,
        "text_length": text_length,
    }


def content_part_without_metadata(part: ContentPart) -> ContentPart:
    """Return a content part with model-visible fields but no host metadata."""

    data = part.to_dict()
    data.pop("metadata", None)
    return ContentPart.from_dict(data)


def tool_call_without_metadata(call: ToolCall) -> ToolCall:
    """Return a tool call with model-visible fields but no host metadata."""

    data = call.to_dict()
    data.pop("metadata", None)
    return ToolCall.from_dict(data)
