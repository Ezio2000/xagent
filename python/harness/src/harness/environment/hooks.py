"""Reusable runtime hooks for controlled runtime tests."""

from __future__ import annotations

from kernel import (
    ModelErrorDecision,
    ModelErrorInfo,
    ModelRequest,
    RuntimeContext,
    RuntimeHook,
)


class RetryModelErrorHook(RuntimeHook):
    """Retry retryable model errors."""

    def on_model_error(
        self,
        error: ModelErrorInfo,
        request: ModelRequest,
        context: RuntimeContext,
    ) -> ModelErrorDecision | None:
        _ = request, context
        return ModelErrorDecision(retry=error.retryable)
