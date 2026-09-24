"""Shared helper for calling the MCP physics server from Strands tools."""
import asyncio
import concurrent.futures
import json
import os

from fastmcp import Client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


def _unwrap_result(result) -> dict:
    """Convert a fastmcp CallToolResult into a plain JSON-serializable dict."""
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except ValueError:
                    return {"result": text}
    return {"result": str(result)}


async def _call(tool_name: str, params: dict) -> dict:
    async with Client(MCP_SERVER_URL) as client:
        return _unwrap_result(await client.call_tool(tool_name, params))


def call_mcp_tool(tool_name: str, params: dict) -> dict:
    """Call an MCP server tool from synchronous tool code.

    Safe whether or not an event loop is already running in this thread:
    Strands may invoke tools from inside its own loop, where a bare
    asyncio.run() raises RuntimeError.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call(tool_name, params))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _call(tool_name, params)).result()
