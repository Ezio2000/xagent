"""Opt-in trace construction and deterministic verification."""

from jharness.kernel.diagnostics.trace import (
    RequestKind,
    RunTrace,
    TraceEntry,
    TraceHeader,
    build_trace,
)
from jharness.kernel.diagnostics.verification import TraceError, TraceVerification, verify_trace

__all__ = [
    "RequestKind",
    "RunTrace",
    "TraceEntry",
    "TraceError",
    "TraceHeader",
    "TraceVerification",
    "build_trace",
    "verify_trace",
]
