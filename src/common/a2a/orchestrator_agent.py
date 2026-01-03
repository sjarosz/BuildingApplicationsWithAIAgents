#!/usr/bin/env python3
"""
orchestrator_agent.py
A2A Orchestrator Agent - the entry point that coordinates with the Planner Agent
and executes tools.

This agent:
- Exposes /.well-known/agent.json (A2A agent card)
- Exposes /api for JSON-RPC requests
- Method: executeTask - receives user tasks, calls planner, executes tools, returns results

Architecture:
    Client → Orchestrator → Planner (A2A) → Orchestrator executes tools → Client

Run with: python -m src.common.a2a.orchestrator_agent

Environment:
- Set OPENAI_API_KEY in your environment or in a .env file
"""

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file if present

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.a2a.a2a_protocol import (
    AgentCard,
    ToolSchema,
    A2AClient,
    make_response,
    make_error_response,
    validate_request,
    extract_method_and_params,
    JSONRPCError,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from src.common.a2a.tools import execute_tool_plan, get_tool_names

# #region agent log - debug instrumentation
import json as _debug_json
_DEBUG_LOG_PATH = "/Users/jarosz/projects/BuildingApplicationsWithAIAgents/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data=None):
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(_debug_json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data or {}, "timestamp": __import__("time").time(), "sessionId": "debug-session", "file": "orchestrator"}) + "\n")
    except: pass
# #endregion

# Safe logging wrapper - doesn't crash when Loki isn't available
def log_to_loki(label: str, message: str):
    """Log to Loki if available, otherwise just print."""
    # #region agent log
    _debug_log("B", "orchestrator:log_to_loki:entry", "log_to_loki called", {"label": label})
    # #endregion
    try:
        print(f"[{label}] {message}")  # Always print to console
        # #region agent log
        _debug_log("B", "orchestrator:log_to_loki:skip_loki", "Skipping Loki call", {})
        # #endregion
    except Exception as e:
        # #region agent log
        _debug_log("B", "orchestrator:log_to_loki:exception", f"Exception: {type(e).__name__}", {"error": str(e)})
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

ORCHESTRATOR_PORT = 8001
ORCHESTRATOR_HOST = "0.0.0.0"
PLANNER_URL = "http://localhost:8002"

# ─── Agent Card (A2A Spec Compliant) ─────────────────────────────────────────

# Raw agent card dict for full A2A spec compliance
AGENT_CARD_DICT = {
    "name": "OrchestratorAgent",
    "description": "Entry point A2A agent that coordinates task execution. Receives tasks via JSON-RPC, delegates planning to a Planner Agent, executes tools, and returns results.",
    "url": f"http://localhost:{ORCHESTRATOR_PORT}",
    "version": "1.0.0",
    "protocol": "A2A/1.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False
    },
    "authentication": {
        "schemes": ["none"]
    },
    "skills": [
        {
            "id": "executeTask",
            "name": "Execute Task",
            "description": "Execute a natural language task by planning and running appropriate tools",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language description of the task to execute"
                    }
                },
                "required": ["task"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "task_results": {
                        "type": "array",
                        "description": "Results from each tool execution",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string"},
                                "params": {"type": "object"},
                                "success": {"type": "boolean"},
                                "result": {},
                                "error": {"type": "string"}
                            }
                        }
                    },
                    "summary": {
                        "type": "string",
                        "description": "Human-readable summary of the task results"
                    },
                    "planning_reasoning": {
                        "type": "string",
                        "description": "Explanation of why tools were selected"
                    },
                    "original_task": {
                        "type": "string",
                        "description": "The original task that was submitted"
                    }
                }
            }
        },
        {
            "id": "listTools",
            "name": "List Available Tools",
            "description": "Returns a list of all tools available for task execution",
            "inputSchema": {
                "type": "object",
                "properties": {}
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of available tool names"
                    }
                }
            }
        }
    ],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "provider": {
        "organization": "BuildingApplicationsWithAIAgents",
        "url": "https://github.com/your-org/agents"
    }
}

# Also create AgentCard object for backward compatibility
agent_card = AgentCard(
    identity="OrchestratorAgent",
    description="Entry point A2A agent that coordinates task execution with the Planner Agent",
    capabilities=["executeTask", "listTools"],
    schemas={
        "executeTask": ToolSchema(
            input={"task": "string"},
            output={"task_results": "array", "summary": "string"}
        ),
        "listTools": ToolSchema(
            input={},
            output={"tools": "array"}
        )
    },
    endpoint=f"http://localhost:{ORCHESTRATOR_PORT}/api",
    auth_methods=["none"],
    version="1.0"
)

# ─── LLM for Summary Generation ──────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orchestrator Agent",
    description="A2A Orchestrator Agent - coordinates task execution"
)


@app.get("/.well-known/agent.json")
async def get_agent_card():
    """Return the full A2A-compliant agent card for discovery."""
    return JSONResponse(content=AGENT_CARD_DICT)


@app.get("/.well-known/agent-card")
async def get_agent_card_alt():
    """Alternative endpoint for agent card (some clients use this path)."""
    return JSONResponse(content=AGENT_CARD_DICT)


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
    
    log_to_loki("orchestrator.agent", f"Received RPC: method={method}, id={request_id}")
    
    # Route to method handler
    if method == "executeTask":
        return await handle_execute_task(params, request_id)
    elif method == "listTools":
        return await handle_list_tools(request_id)
    else:
        return JSONResponse(
            content=make_error_response(METHOD_NOT_FOUND, f"Method not found: {method}", request_id),
            status_code=400
        )


async def handle_list_tools(request_id: str) -> JSONResponse:
    """Handle the listTools method - returns available tools."""
    tools = get_tool_names()
    return JSONResponse(content=make_response({"tools": tools}, request_id))


async def handle_execute_task(params: Dict[str, Any], request_id: str) -> JSONResponse:
    """
    Handle the executeTask method.
    
    1. Validates the task parameter
    2. Calls the Planner Agent to get a tool execution plan
    3. Executes the planned tools
    4. Generates a summary of results
    5. Returns the complete response
    """
    with tracer.start_as_current_span("orchestrator.execute_task") as span:
        task = params.get("task")
        
        if not task:
            return JSONResponse(
                content=make_error_response(INVALID_PARAMS, "Missing required parameter: task", request_id),
                status_code=400
            )
        
        span.set_attribute("task", task)
        log_to_loki("orchestrator.agent", f"Executing task: {task}")
        
        try:
            # Step 1: Call the Planner Agent to get tool plan
            with tracer.start_as_current_span("orchestrator.call_planner"):
                log_to_loki("orchestrator.agent", f"Calling Planner Agent at {PLANNER_URL}")
                
                with A2AClient(PLANNER_URL) as planner:
                    plan_result = planner.call("planToolUsage", {"task": task})
                
                tools_to_execute = plan_result.get("tools", [])
                reasoning = plan_result.get("reasoning", "")
                
                log_to_loki("orchestrator.agent", f"Planner returned {len(tools_to_execute)} tools to execute")
                span.set_attribute("planned_tools_count", len(tools_to_execute))
            
            # Step 2: Execute the planned tools
            with tracer.start_as_current_span("orchestrator.execute_tools"):
                if tools_to_execute:
                    log_to_loki("orchestrator.agent", f"Executing tools: {[t['name'] for t in tools_to_execute]}")
                    task_results = execute_tool_plan(tools_to_execute)
                else:
                    log_to_loki("orchestrator.agent", "No tools to execute")
                    task_results = []
                
                span.set_attribute("executed_tools_count", len(task_results))
            
            # Step 3: Generate a human-readable summary
            with tracer.start_as_current_span("orchestrator.generate_summary"):
                summary = await generate_summary(task, task_results, reasoning)
            
            # Build the response
            result = {
                "task_results": task_results,
                "summary": summary,
                "planning_reasoning": reasoning,
                "original_task": task
            }
            
            log_to_loki("orchestrator.agent", f"Task completed successfully")
            return JSONResponse(content=make_response(result, request_id))
            
        except JSONRPCError as e:
            log_to_loki("orchestrator.agent", f"A2A call failed: {e}")
            return JSONResponse(
                content=make_error_response(e.code, f"Planner error: {e.message}", request_id),
                status_code=500
            )
        except Exception as e:
            log_to_loki("orchestrator.agent", f"Error executing task: {e}")
            return JSONResponse(
                content=make_error_response(INTERNAL_ERROR, str(e), request_id),
                status_code=500
            )


async def generate_summary(task: str, task_results: List[Dict[str, Any]], reasoning: str) -> str:
    """
    Generate a human-readable summary of the task results using LLM.
    
    Args:
        task: The original task
        task_results: Results from tool executions
        reasoning: The planner's reasoning
        
    Returns:
        A human-readable summary string
    """
    if not task_results:
        return "No tools were needed to complete this task."
    
    # Check if all tools succeeded
    all_succeeded = all(r.get("success", False) for r in task_results)
    
    if not all_succeeded:
        failed = [r for r in task_results if not r.get("success", False)]
        return f"Task partially completed. {len(failed)} tool(s) failed: " + \
               ", ".join(f"{f['tool']}: {f.get('error', 'unknown error')}" for f in failed)
    
    # Use LLM to generate a nice summary
    try:
        results_text = "\n".join(
            f"- {r['tool']}: {r['result']}" for r in task_results
        )
        
        messages = [
            SystemMessage(content="You are a helpful assistant. Summarize the results of the tool executions in a clear, concise sentence or two."),
            HumanMessage(content=f"Original task: {task}\n\nTool results:\n{results_text}\n\nProvide a brief summary:")
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        # Fallback to simple summary
        log_to_loki("orchestrator.agent", f"Summary generation failed, using fallback: {e}")
        return f"Completed {len(task_results)} tool(s): " + \
               ", ".join(f"{r['tool']}={r['result']}" for r in task_results)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "OrchestratorAgent"}


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting Orchestrator Agent on http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}")
    print(f"Agent card available at: http://localhost:{ORCHESTRATOR_PORT}/.well-known/agent.json")
    print(f"API endpoint: http://localhost:{ORCHESTRATOR_PORT}/api")
    print(f"Planner Agent expected at: {PLANNER_URL}")
    print("\nMake sure to start the Planner Agent first:")
    print(f"  python -m src.common.a2a.planner_agent")
    uvicorn.run(app, host=ORCHESTRATOR_HOST, port=ORCHESTRATOR_PORT)

