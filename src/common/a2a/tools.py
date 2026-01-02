"""
tools.py
Tool definitions for the A2A Orchestrator-Planner system.

Provides:
- Calculator tools: add, subtract, multiply, divide
- Generic tools: get_current_time, echo, format_text
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    result: Any
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"success": self.success, "result": self.result}
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class ToolDefinition:
    """Definition of a tool including its metadata and implementation."""
    name: str
    description: str
    parameters: Dict[str, str]  # param_name -> type description
    function: Callable[..., Any]
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to schema format for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


# ─── Calculator Tools ────────────────────────────────────────────────────────

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# ─── Generic Tools ───────────────────────────────────────────────────────────

def get_current_time() -> str:
    """Get the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def echo(message: str) -> str:
    """Echo back the provided message."""
    return message


def format_text(text: str, style: str = "uppercase") -> str:
    """
    Format text according to the specified style.
    
    Styles: uppercase, lowercase, title, reverse
    """
    style = style.lower()
    if style == "uppercase":
        return text.upper()
    elif style == "lowercase":
        return text.lower()
    elif style == "title":
        return text.title()
    elif style == "reverse":
        return text[::-1]
    else:
        raise ValueError(f"Unknown style: {style}. Use: uppercase, lowercase, title, reverse")


# ─── Tool Registry ───────────────────────────────────────────────────────────

TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    # Calculator tools
    "add": ToolDefinition(
        name="add",
        description="Add two numbers together",
        parameters={"a": "number", "b": "number"},
        function=add
    ),
    "subtract": ToolDefinition(
        name="subtract",
        description="Subtract b from a (returns a - b)",
        parameters={"a": "number", "b": "number"},
        function=subtract
    ),
    "multiply": ToolDefinition(
        name="multiply",
        description="Multiply two numbers together",
        parameters={"a": "number", "b": "number"},
        function=multiply
    ),
    "divide": ToolDefinition(
        name="divide",
        description="Divide a by b (returns a / b). Cannot divide by zero.",
        parameters={"a": "number", "b": "number"},
        function=divide
    ),
    
    # Generic tools
    "get_current_time": ToolDefinition(
        name="get_current_time",
        description="Get the current UTC time in ISO format",
        parameters={},
        function=get_current_time
    ),
    "echo": ToolDefinition(
        name="echo",
        description="Echo back the provided message",
        parameters={"message": "string"},
        function=echo
    ),
    "format_text": ToolDefinition(
        name="format_text",
        description="Format text according to a style (uppercase, lowercase, title, reverse)",
        parameters={"text": "string", "style": "string (optional, default: uppercase)"},
        function=format_text
    ),
}


def get_available_tools() -> List[Dict[str, Any]]:
    """Get list of all available tools with their schemas."""
    return [tool.to_schema() for tool in TOOL_DEFINITIONS.values()]


def get_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return list(TOOL_DEFINITIONS.keys())


def execute_tool(name: str, params: Dict[str, Any]) -> ToolResult:
    """
    Execute a tool by name with the given parameters.
    
    Args:
        name: Name of the tool to execute
        params: Parameters to pass to the tool
        
    Returns:
        ToolResult with success status and result/error
    """
    if name not in TOOL_DEFINITIONS:
        return ToolResult(
            success=False,
            result=None,
            error=f"Unknown tool: {name}. Available tools: {', '.join(get_tool_names())}"
        )
    
    tool = TOOL_DEFINITIONS[name]
    
    try:
        result = tool.function(**params)
        return ToolResult(success=True, result=result)
    except TypeError as e:
        return ToolResult(
            success=False,
            result=None,
            error=f"Invalid parameters for {name}: {e}"
        )
    except Exception as e:
        return ToolResult(
            success=False,
            result=None,
            error=f"Tool execution error: {e}"
        )


def execute_tool_plan(tool_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute a list of planned tool calls.
    
    Args:
        tool_plan: List of {"name": str, "params": dict} objects
        
    Returns:
        List of results with tool name and execution result
    """
    results = []
    
    for tool_call in tool_plan:
        name = tool_call.get("name", "")
        params = tool_call.get("params", {})
        
        result = execute_tool(name, params)
        
        results.append({
            "tool": name,
            "params": params,
            "success": result.success,
            "result": result.result,
            "error": result.error
        })
    
    return results


# ─── Tool Description for LLM ────────────────────────────────────────────────

def get_tools_description_for_llm() -> str:
    """
    Generate a formatted description of all tools for LLM consumption.
    
    Returns:
        A string describing all available tools
    """
    lines = ["Available tools:\n"]
    
    for name, tool in TOOL_DEFINITIONS.items():
        params_str = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items()) if tool.parameters else "none"
        lines.append(f"- {name}: {tool.description}")
        lines.append(f"  Parameters: {params_str}")
        lines.append("")
    
    return "\n".join(lines)

