"""JSONL logging for mala agent sessions.

Provides Claude Code style JSONL logging for full message logging,
including tool use/results, text blocks, and session metadata.
"""

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from ..tools.env import JSONL_LOG_DIR


def get_git_branch(cwd: Path) -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


class JSONLLogger:
    """Claude Code style JSONL logger for full message logging."""

    def __init__(self, session_id: str, cwd: Path):
        self.session_id = session_id
        self.cwd = cwd
        self.git_branch = get_git_branch(cwd)
        JSONL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = JSONL_LOG_DIR / f"{session_id}.jsonl"
        self.file = open(self.log_path, "w")
        self.parent_uuid: str | None = None
        self.message_count = 0

    def _write(self, entry: dict[str, Any]):
        """Write a JSON entry as a line."""
        self.file.write(json.dumps(entry, default=str) + "\n")
        self.file.flush()

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 UTC format with milliseconds."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def log_message(self, message: Any, message_type: str = "assistant"):
        """Log a full message from the SDK.

        Follows Claude Code JSONL format:
        - Assistant messages contain: text, tool_use, thinking blocks
        - User messages contain: tool_result blocks (logged separately)
        """
        msg_uuid = str(uuid.uuid4())
        timestamp = self._get_timestamp()

        base_entry: dict[str, Any] = {
            "uuid": msg_uuid,
            "parentUuid": self.parent_uuid,
            "type": message_type,
            "timestamp": timestamp,
            "sessionId": self.session_id,
            "cwd": str(self.cwd),
            "gitBranch": self.git_branch,
        }

        # Handle different message types from SDK
        if isinstance(message, AssistantMessage):
            assistant_content = []
            tool_results = []

            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": getattr(block, "id", f"tool_{self.message_count}"),
                            "name": block.name,
                            "input": block.input,  # Full params, not truncated
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    # Tool results are logged as separate "user" type entries
                    tool_use_id = getattr(block, "tool_use_id", None)
                    is_error = getattr(block, "is_error", False)
                    tool_result_entry: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": block.content,
                    }
                    if is_error:
                        tool_result_entry["is_error"] = True
                    tool_results.append(tool_result_entry)

            # Log assistant content if present
            if assistant_content:
                msg_id = getattr(message, "id", None) or f"msg_{self.message_count}"
                entry = base_entry.copy()
                entry["message"] = {
                    "id": msg_id,
                    "content": assistant_content,
                }
                self._write(entry)
                self.parent_uuid = msg_uuid
                self.message_count += 1

            # Log tool results as separate "user" type entries
            for tool_result in tool_results:
                result_uuid = str(uuid.uuid4())
                result_entry = {
                    "uuid": result_uuid,
                    "parentUuid": self.parent_uuid,
                    "type": "user",
                    "timestamp": self._get_timestamp(),
                    "sessionId": self.session_id,
                    "cwd": str(self.cwd),
                    "gitBranch": self.git_branch,
                    "message": {
                        "content": [tool_result],
                    },
                }
                self._write(result_entry)
                self.parent_uuid = result_uuid
                self.message_count += 1

            return  # Already handled writing

        elif isinstance(message, ResultMessage):
            # ResultMessage maps to turn_end in Claude Code spec
            base_entry["type"] = "turn_end"
            base_entry["message"] = {
                "result": message.result,
            }
        else:
            # Raw message
            base_entry["message"] = message

        self._write(base_entry)
        self.parent_uuid = msg_uuid
        self.message_count += 1

    def log_user_prompt(self, prompt: str):
        """Log the initial user prompt."""
        msg_uuid = str(uuid.uuid4())
        timestamp = self._get_timestamp()

        entry = {
            "uuid": msg_uuid,
            "parentUuid": None,
            "type": "user",
            "timestamp": timestamp,
            "sessionId": self.session_id,
            "cwd": str(self.cwd),
            "gitBranch": self.git_branch,
            "message": {
                "content": [{"type": "text", "text": prompt}],
            },
        }
        self._write(entry)
        self.parent_uuid = msg_uuid
        self.message_count += 1

    def close(self):
        self.file.close()
