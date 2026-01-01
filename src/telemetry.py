"""Backward-compatibility shim for src.telemetry.

This module re-exports all public symbols from src.infra.telemetry.
New code should import directly from src.infra.telemetry.
"""

from src.infra.telemetry import (
    NullSpan,
    NullTelemetryProvider,
    TelemetryProvider,
    TelemetrySpan,
)

__all__ = ["NullSpan", "NullTelemetryProvider", "TelemetryProvider", "TelemetrySpan"]
