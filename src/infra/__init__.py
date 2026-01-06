"""Infrastructure layer for mala.

This package contains all infrastructure concerns:
- clients/: External service clients (Anthropic, Beads CLI, Cerberus)
- io/: I/O utilities (config, event sink, logging, session log parsing)
- tools/: Command execution, environment, and locking utilities
- hooks/: Claude Agent SDK hooks (security, caching, locking)

Infrastructure modules should not depend on:
- CLI layer (src.cli, src.main)
- Orchestration layer (src.orchestrator, src.orchestrator_factory, src.orchestrator_types)
- Pipeline layer (src.pipeline)

They may depend on:
- Core layer (src.core.models, src.core.protocols, src.core.log_events)
- Domain layer for shared types (but not business logic)
"""
