"""Message and content protocol types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

Role = Literal["system", "user", "assistant", "tool"]
KNOWN_ROLES = {"system", "user", "assistant", "tool"}
PartType = str


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _empty_parts() -> list[ContentPart]:
    return []


def _empty_tool_calls() -> list[ToolCall]:
    return []


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _copy_extra(value: Mapping[str, Any] | None, reserved: set[str], label: str) -> dict[str, Any]:
    extra = _copy_mapping(value)
    conflicts = reserved & extra.keys()
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"{label} extra cannot override reserved field(s): {names}")
    return extra


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
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("part type must be a non-empty string")
        if self.type == "text" and self.text is None:
            raise ValueError("text parts require text")
        if self.uri is not None and self.ref is not None:
            raise ValueError("content part cannot set both uri and ref")
        self.data = _copy_mapping(self.data)
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {"type", "text", "uri", "ref", "media_type", "name", "data", "metadata"},
            "content part",
        )

    @classmethod
    def text_part(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(type="text", text=text, metadata=metadata or {}, extra=extra or {})

    @classmethod
    def image_uri(
        cls,
        uri: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="image",
            uri=uri,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
            extra=extra or {},
        )

    @classmethod
    def image_ref(
        cls,
        ref: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="image",
            ref=ref,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
            extra=extra or {},
        )

    @classmethod
    def file_uri(
        cls,
        uri: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="file",
            uri=uri,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
            extra=extra or {},
        )

    @classmethod
    def file_ref(
        cls,
        ref: str,
        *,
        media_type: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="file",
            ref=ref,
            media_type=media_type,
            name=name,
            metadata=metadata or {},
            extra=extra or {},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContentPart:
        known = {"type", "text", "uri", "ref", "media_type", "name", "data", "metadata"}
        return cls(
            type=str(value["type"]),
            text=cast(str | None, value.get("text")),
            uri=cast(str | None, value.get("uri")),
            ref=cast(str | None, value.get("ref")),
            media_type=cast(str | None, value.get("media_type")),
            name=cast(str | None, value.get("name")),
            data=_expect_mapping(value.get("data") or {}, "content part data"),
            metadata=_expect_mapping(value.get("metadata") or {}, "content part metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
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
        data.update(
            _copy_extra(
                self.extra,
                {"type", "text", "uri", "ref", "media_type", "name", "data", "metadata"},
                "content part",
            )
        )
        return data


@dataclass(slots=True)
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tool call id must not be empty")
        if not self.name:
            raise ValueError("tool call name must not be empty")
        self.arguments = _copy_mapping(self.arguments)
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {"id", "name", "arguments", "metadata"},
            "tool call",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolCall:
        known = {"id", "name", "arguments", "metadata"}
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            arguments=_expect_mapping(value.get("arguments") or {}, "tool call arguments"),
            metadata=_expect_mapping(value.get("metadata") or {}, "tool call metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": _copy_mapping(self.arguments),
        }
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {"id", "name", "arguments", "metadata"},
                "tool call",
            )
        )
        return data


@dataclass(slots=True)
class Message:
    """A conversation message used by the runtime."""

    role: Role
    parts: list[ContentPart] = field(default_factory=_empty_parts)
    tool_calls: list[ToolCall] = field(default_factory=_empty_tool_calls)
    tool_call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
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
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {"role", "parts", "tool_call_id", "tool_calls", "metadata"},
            "message",
        )

    @classmethod
    def system(
        cls, parts: Sequence[ContentPart], *, metadata: Mapping[str, Any] | None = None
    ) -> Message:
        return cls(role="system", parts=list(parts), metadata=metadata or {})

    @classmethod
    def system_text(cls, text: str, *, metadata: Mapping[str, Any] | None = None) -> Message:
        return cls.system([ContentPart.text_part(text)], metadata=metadata)

    @classmethod
    def user(
        cls, parts: Sequence[ContentPart], *, metadata: Mapping[str, Any] | None = None
    ) -> Message:
        return cls(role="user", parts=list(parts), metadata=metadata or {})

    @classmethod
    def user_text(cls, text: str, *, metadata: Mapping[str, Any] | None = None) -> Message:
        return cls.user([ContentPart.text_part(text)], metadata=metadata)

    @classmethod
    def assistant(
        cls,
        parts: Sequence[ContentPart],
        tool_calls: Sequence[ToolCall] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            role="assistant",
            parts=list(parts),
            tool_calls=list(tool_calls or ()),
            metadata=metadata or {},
            extra=extra or {},
        )

    @classmethod
    def assistant_text(
        cls,
        text: str,
        tool_calls: Sequence[ToolCall] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls.assistant(
            [ContentPart.text_part(text)],
            tool_calls,
            metadata=metadata,
            extra=extra,
        )

    @classmethod
    def tool(
        cls,
        parts: Sequence[ContentPart],
        tool_call_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            role="tool",
            parts=list(parts),
            tool_call_id=tool_call_id,
            metadata=metadata or {},
            extra=extra or {},
        )

    @classmethod
    def tool_text(
        cls,
        text: str,
        tool_call_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls.tool(
            [ContentPart.text_part(text)],
            tool_call_id,
            metadata=metadata,
            extra=extra,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        known = {"role", "parts", "tool_call_id", "tool_calls", "metadata"}
        role = str(value["role"])
        if role not in KNOWN_ROLES:
            raise ValueError(f"unsupported message role: {role}")
        return cls(
            role=cast(Role, role),
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "message part"))
                for part in cast(Sequence[object], value.get("parts") or ())
            ],
            tool_calls=[
                ToolCall.from_dict(_expect_mapping(call, "message tool call"))
                for call in cast(Sequence[object], value.get("tool_calls") or ())
            ],
            tool_call_id=cast(str | None, value.get("tool_call_id")),
            metadata=_expect_mapping(value.get("metadata") or {}, "message metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
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
        data.update(
            _copy_extra(
                self.extra,
                {"role", "parts", "tool_call_id", "tool_calls", "metadata"},
                "message",
            )
        )
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
