#!/usr/bin/env python3
"""Mock MCP server for testing MorphLLM MCP tool integration.

This server mimics the MorphLLM MCP server's tools (edit_file, warpgrep_codebase_search)
without making real API calls. It provides working implementations for E2E tests.

Usage: python mock_mcp_server.py
"""

import json
import sys
from pathlib import Path


def send_response(response: dict) -> None:
    """Send a JSON-RPC response to stdout."""
    msg = json.dumps(response)
    # MCP uses Content-Length header framing
    sys.stdout.write(f"Content-Length: {len(msg)}\r\n\r\n{msg}")
    sys.stdout.flush()


def read_message() -> dict | None:
    """Read a JSON-RPC message from stdin."""
    # Read Content-Length header
    line = sys.stdin.readline()
    if not line:
        return None

    # Skip empty lines
    while line.strip() == "":
        line = sys.stdin.readline()
        if not line:
            return None

    # Parse Content-Length
    if line.startswith("Content-Length:"):
        length = int(line.split(":")[1].strip())
        # Read empty line after header
        sys.stdin.readline()
        # Read content
        content = sys.stdin.read(length)
        return json.loads(content)

    # Try parsing as raw JSON (some clients don't use framing)
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def handle_initialize(request_id: int) -> dict:
    """Handle initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock-morphllm", "version": "1.0.0"},
        },
    }


def handle_tools_list(request_id: int) -> dict:
    """Handle tools/list request."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": [
                {
                    "name": "edit_file",
                    "description": "Edit a file by applying code changes",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to edit",
                            },
                            "code_edit": {
                                "type": "string",
                                "description": "The code edit to apply",
                            },
                        },
                        "required": ["path", "code_edit"],
                    },
                },
                {
                    "name": "warpgrep_codebase_search",
                    "description": "Search codebase for patterns",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query/pattern",
                            }
                        },
                        "required": ["query"],
                    },
                },
            ]
        },
    }


def handle_tool_call(request_id: int, tool_name: str, arguments: dict) -> dict:
    """Handle tools/call request."""
    if tool_name == "edit_file":
        return handle_edit_file(request_id, arguments)
    elif tool_name == "warpgrep_codebase_search":
        return handle_warpgrep(request_id, arguments)
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }


def handle_edit_file(request_id: int, arguments: dict) -> dict:
    """Handle edit_file tool call - actually edit the file."""
    path = arguments.get("path", "")
    code_edit = arguments.get("code_edit", "")

    try:
        file_path = Path(path)

        # Simple edit: if code_edit contains the new content, write it
        # For more complex edits, append or replace based on content
        if file_path.exists():
            current = file_path.read_text()
            # If code_edit looks like a replacement (contains old -> new pattern)
            if "->" in code_edit or "change" in code_edit.lower():
                # Try to parse as "old -> new" or just apply the edit
                if "a - b" in current and ("a + b" in code_edit or "+ b" in code_edit):
                    new_content = current.replace("a - b", "a + b")
                    file_path.write_text(new_content)
                elif "# edited by morph" in code_edit or "edited" in code_edit.lower():
                    file_path.write_text(current.rstrip() + "\n# edited by morph\n")
                else:
                    file_path.write_text(current + "\n" + code_edit)
            else:
                # Append the edit
                file_path.write_text(current.rstrip() + "\n" + code_edit + "\n")
        else:
            file_path.write_text(code_edit)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"Successfully edited {path}"}]
            },
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"Error editing file: {e}"}],
                "isError": True,
            },
        }


def handle_warpgrep(request_id: int, arguments: dict) -> dict:
    """Handle warpgrep_codebase_search tool call - search for patterns."""
    query = arguments.get("query", "")

    # Get the cwd from environment or use current dir
    import os

    cwd = Path(os.environ.get("WORKSPACE_PATH", "."))

    results = []
    try:
        for py_file in cwd.rglob("*.py"):
            try:
                content = py_file.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if query in line:
                        results.append(f"{py_file}:{i}: {line.strip()}")
            except Exception:
                continue
    except Exception:
        pass

    if results:
        result_text = "\n".join(results[:20])  # Limit results
    else:
        result_text = f"No matches found for: {query}"

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": result_text}]},
    }


def main() -> None:
    """Main MCP server loop."""
    # Send initialized notification capability
    while True:
        msg = read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        request_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send_response(handle_initialize(request_id))
        elif method == "notifications/initialized":
            # No response needed for notifications
            pass
        elif method == "tools/list":
            send_response(handle_tools_list(request_id))
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            send_response(handle_tool_call(request_id, tool_name, arguments))
        elif request_id is not None:
            # Unknown method with id - send error
            send_response(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )


if __name__ == "__main__":
    main()
