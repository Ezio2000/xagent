"""Tool fixtures and registry doubles for controlled runtime scenarios."""

from harness.tools.registry import (
    HarnessToolRegistry,
    RecordingToolRegistry,
    ScriptedToolRegistry,
    ToolInvocationRecord,
)
from harness.tools.toolkit_fixtures import (
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
    "HarnessToolRegistry",
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
