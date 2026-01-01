"""mala: Agent SDK orchestrator for parallel issue processing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestration.orchestrator import MalaOrchestrator

__version__ = "0.1.0"
__all__ = ["MalaOrchestrator", "__version__"]


def __getattr__(name: str) -> type["MalaOrchestrator"]:
    """Lazy load MalaOrchestrator to defer SDK imports."""
    if name == "MalaOrchestrator":
        from .orchestration.orchestrator import MalaOrchestrator

        return MalaOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
