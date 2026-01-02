#!/usr/bin/env python3
"""
planner_agent.py
A2A Planner Agent that uses GPT-4o to determine which tools to use for a given task.

This agent:
- Exposes /.well-known/agent.json (A2A agent card)
- Exposes /api for JSON-RPC requests
- Method: planToolUsage - analyzes a task and returns a tool execution plan

Run with: python -m src.common.a2a.planner_agent

Environment:
- Set OPENAI_API_KEY in your environment or in a .env file
"""

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file if present

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.a2a.a2a_protocol import (
    AgentCard,
    ToolSchema,
    make_response,
    make_error_response,
    validate_request,
    extract_method_and_params,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from src.common.a2a.tools import get_tools_description_for_llm, get_tool_names

# #region agent log - debug instrumentation
import json as _debug_json
_DEBUG_LOG_PATH = "/Users/jarosz/projects/BuildingApplicationsWithAIAgents/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data=None):
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(_debug_json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data or {}, "timestamp": __import__("time").time(), "sessionId": "debug-session", "file": "planner"}) + "\n")
    except: pass
# #endregion

# Safe logging wrapper - doesn't crash when Loki isn't available
def log_to_loki(label: str, message: str):
    """Log to Loki if available, otherwise just print."""
    # #region agent log
    _debug_log("B", "planner:log_to_loki:entry", "log_to_loki called", {"label": label})
    # #endregion
    try:
        print(f"[{label}] {message}")  # Always print to console
        # #region agent log
        _debug_log("B", "planner:log_to_loki:skip_loki", "Skipping Loki call", {})
        # #endregion
    except Exception as e:
        # #region agent log
        _debug_log("B", "planner:log_to_loki:exception", f"Exception: {type(e).__name__}", {"error": str(e)})
        # #endregion
        print(f"[{label}] {message}")

# Safe tracer - provides a no-op tracer if Tempo isn't available
class NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, key, value): pass

class NoOpTracer:
    def start_as_current_span(self, name):
        return NoOpSpan()

try:
    from src.common.observability.instrument_tempo import tracer
except Exception:
    tracer = NoOpTracer()

# ─── Configuration ───────────────────────────────────────────────────────────

PLANNER_PORT = 8002
PLANNER_HOST = "0.0.0.0"

# ─── Agent Card ──────────────────────────────────────────────────────────────

agent_card = AgentCard(
    identity="PlannerAgent",
    description="LLM-based planner that determines which tools to use for a given task",
    capabilities=["planToolUsage"],
    schemas={
        "planToolUsage": ToolSchema(
            input={"task": "string - The task description to plan tools for"},
            output={"tools": "array - List of {name, params} tool calls to execute"}
        )
    },
    endpoint=f"http://localhost:{PLANNER_PORT}/api",
    auth_methods=["none"],
    version="1.0"
)

# ─── LLM Setup ───────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

SYSTEM_PROMPT = """You are a tool planning assistant. Given a user's task, you determine which tools to call and with what parameters.

{tools_description}

INSTRUCTIONS:
1. Analyze the user's task carefully
2. Determine which tools are needed to complete the task
3. Return a JSON object with a "tools" array containing the tool calls
4. Each tool call should have "name" (tool name) and "params" (parameter object)
5. If no tools are needed, return an empty tools array
6. Only use tools that are available in the list above

RESPONSE FORMAT (JSON only, no markdown):
{{
  "tools": [
    {{"name": "tool_name", "params": {{"param1": value1, "param2": value2}}}},
    ...
  ],
  "reasoning": "Brief explanation of why these tools were chosen"
}}

Examples:
- Task: "What is 5 + 3?" → {{"tools": [{{"name": "add", "params": {{"a": 5, "b": 3}}}}], "reasoning": "Simple addition"}}
- Task: "Multiply 10 by 5 and then tell me the time" → {{"tools": [{{"name": "multiply", "params": {{"a": 10, "b": 5}}}}, {{"name": "get_current_time", "params": {{}}}}], "reasoning": "Two operations needed"}}
"""

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Planner Agent", description="A2A Planner Agent with LLM-based tool planning")


@app.get("/.well-known/agent.json")
async def get_agent_card():
    """Return the agent card for A2A discovery."""
    return JSONResponse(content=agent_card.to_dict())


@app.post("/api")
async def handle_rpc(request: Request):
    """Handle JSON-RPC requests."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            content=make_error_response(-32700, "Parse error"),
            status_code=400
        )
    
    if not validate_request(body):
        return JSONResponse(
            content=make_error_response(-32600, "Invalid Request", body.get("id")),
            status_code=400
        )
    
    method, params, request_id = extract_method_and_params(body)
    
    log_to_loki("planner.agent", f"Received RPC: method={method}, id={request_id}")
    
    # Route to method handler
    if method == "planToolUsage":
        return await handle_plan_tool_usage(params, request_id)
    else:
        return JSONResponse(
            content=make_error_response(METHOD_NOT_FOUND, f"Method not found: {method}", request_id),
            status_code=400
        )


async def handle_plan_tool_usage(params: Dict[str, Any], request_id: str) -> JSONResponse:
    """
    Handle the planToolUsage method.
    
    Uses GPT-4o to analyze the task and determine which tools to call.
    """
    with tracer.start_as_current_span("planner.plan_tool_usage") as span:
        task = params.get("task")
        
        if not task:
            return JSONResponse(
                content=make_error_response(INVALID_PARAMS, "Missing required parameter: task", request_id),
                status_code=400
            )
        
        span.set_attribute("task", task)
        log_to_loki("planner.agent", f"Planning tools for task: {task}")
        
        try:
            # Build the prompt with available tools
            tools_description = get_tools_description_for_llm()
            system_message = SYSTEM_PROMPT.format(tools_description=tools_description)
            
            # Call LLM for tool planning
            with tracer.start_as_current_span("planner.llm_call"):
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=f"Task: {task}")
                ]
                
                response = llm.invoke(messages)
                raw_response = response.content.strip()
            
            log_to_loki("planner.agent", f"LLM response: {raw_response[:200]}...")
            
            # Parse the LLM response
            try:
                # Handle potential markdown code blocks
                if raw_response.startswith("```"):
                    # Extract JSON from code block
                    lines = raw_response.split("\n")
                    json_lines = []
                    in_block = False
                    for line in lines:
                        if line.startswith("```"):
                            in_block = not in_block
                            continue
                        if in_block:
                            json_lines.append(line)
                    raw_response = "\n".join(json_lines)
                
                plan = json.loads(raw_response)
            except json.JSONDecodeError as e:
                log_to_loki("planner.agent", f"Failed to parse LLM response as JSON: {e}")
                return JSONResponse(
                    content=make_error_response(INTERNAL_ERROR, f"Failed to parse LLM response: {e}", request_id),
                    status_code=500
                )
            
            # Validate the plan structure
            if "tools" not in plan:
                plan = {"tools": [], "reasoning": "No tools identified"}
            
            # Validate tool names
            available_tools = get_tool_names()
            validated_tools = []
            for tool_call in plan.get("tools", []):
                if tool_call.get("name") in available_tools:
                    validated_tools.append(tool_call)
                else:
                    log_to_loki("planner.agent", f"Skipping unknown tool: {tool_call.get('name')}")
            
            result = {
                "tools": validated_tools,
                "reasoning": plan.get("reasoning", ""),
                "original_task": task
            }
            
            span.set_attribute("planned_tools_count", len(validated_tools))
            log_to_loki("planner.agent", f"Planned {len(validated_tools)} tools for task")
            
            return JSONResponse(content=make_response(result, request_id))
            
        except Exception as e:
            log_to_loki("planner.agent", f"Error planning tools: {e}")
            return JSONResponse(
                content=make_error_response(INTERNAL_ERROR, str(e), request_id),
                status_code=500
            )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "PlannerAgent"}


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting Planner Agent on http://{PLANNER_HOST}:{PLANNER_PORT}")
    print(f"Agent card available at: http://localhost:{PLANNER_PORT}/.well-known/agent.json")
    print(f"API endpoint: http://localhost:{PLANNER_PORT}/api")
    uvicorn.run(app, host=PLANNER_HOST, port=PLANNER_PORT)

