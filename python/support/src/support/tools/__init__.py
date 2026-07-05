"""Tool fixtures and registry doubles for controlled runtime scenarios."""

from support.tools.registry import (
    FixtureToolRegistry,
    RecordingToolRegistry,
    ScriptedToolRegistry,
    ToolInvocationRecord,
)
from support.tools.toolkit_fixtures import (
    AcceptFixtureTool,
    AcceptingWebSearchTool,
    CustomHandoffTool,
    DelayedEchoFixtureTool,
    EchoFixtureTool,
    FailingAcceptTool,
    FailingCustomHandoffTool,
    FailingFixtureTool,
    HandoffFixtureTool,
    ParallelWaitFixtureTool,
    ProgressFixtureTool,
    RejectingWebSearchTool,
    StrictCountFixtureTool,
    StrictCustomHandoffTool,
    WaitFixtureTool,
)

__all__ = [
    "AcceptFixtureTool",
    "AcceptingWebSearchTool",
    "CustomHandoffTool",
    "DelayedEchoFixtureTool",
    "EchoFixtureTool",
    "FailingAcceptTool",
    "FailingCustomHandoffTool",
    "FailingFixtureTool",
    "HandoffFixtureTool",
    "FixtureToolRegistry",
    "ParallelWaitFixtureTool",
    "ProgressFixtureTool",
    "RecordingToolRegistry",
    "RejectingWebSearchTool",
    "ScriptedToolRegistry",
    "StrictCountFixtureTool",
    "StrictCustomHandoffTool",
    "ToolInvocationRecord",
    "WaitFixtureTool",
]
