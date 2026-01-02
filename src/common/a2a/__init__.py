"""
A2A (Agent-to-Agent) Protocol Implementation

This module provides:
- a2a_protocol: JSON-RPC helpers and A2A client utilities
- tools: Calculator and generic tools for agent use
- planner_agent: LLM-based tool planning agent
- orchestrator_agent: Entry point agent that coordinates task execution
"""

from src.common.a2a.a2a_protocol import (
    AgentCard,
    ToolSchema,
    A2AClient,
    AsyncA2AClient,
    JSONRPCError,
    make_request,
    make_response,
    make_error_response,
    validate_request,
    extract_method_and_params,
)

from src.common.a2a.tools import (
    ToolResult,
    ToolDefinition,
    TOOL_DEFINITIONS,
    get_available_tools,
    get_tool_names,
    execute_tool,
    execute_tool_plan,
    get_tools_description_for_llm,
)

__all__ = [
    # Protocol
    "AgentCard",
    "ToolSchema", 
    "A2AClient",
    "AsyncA2AClient",
    "JSONRPCError",
    "make_request",
    "make_response",
    "make_error_response",
    "validate_request",
    "extract_method_and_params",
    # Tools
    "ToolResult",
    "ToolDefinition",
    "TOOL_DEFINITIONS",
    "get_available_tools",
    "get_tool_names",
    "execute_tool",
    "execute_tool_plan",
    "get_tools_description_for_llm",
]

