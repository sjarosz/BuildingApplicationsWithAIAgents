"""
a2a_protocol.py
Shared A2A (Agent-to-Agent) protocol helpers for JSON-RPC communication.

Provides:
- JSON-RPC request/response builders
- A2A client for making agent-to-agent calls
- Agent card schema and utilities
"""

import json
import uuid
import httpx
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# #region agent log - debug instrumentation
import json as _debug_json
_DEBUG_LOG_PATH = "/Users/jarosz/projects/BuildingApplicationsWithAIAgents/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data=None):
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(_debug_json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data or {}, "timestamp": __import__("time").time(), "sessionId": "debug-session", "file": "a2a_protocol"}) + "\n")
    except: pass
# #endregion

# Safe logging wrapper - doesn't crash when Loki isn't available
def log_to_loki(label: str, message: str):
    """Log to Loki if available, otherwise just print."""
    # #region agent log
    _debug_log("D", "a2a_protocol:log_to_loki:entry", "log_to_loki called", {"label": label})
    # #endregion
    try:
        print(f"[{label}] {message}")
        # #region agent log
        _debug_log("D", "a2a_protocol:log_to_loki:success", "Logged to console", {})
        # #endregion
    except Exception as e:
        # #region agent log
        _debug_log("D", "a2a_protocol:log_to_loki:exception", f"Exception: {type(e).__name__}", {"error": str(e)})
        # #endregion
        pass


# ─── JSON-RPC Protocol ───────────────────────────────────────────────────────

class JSONRPCError(Exception):
    """Exception for JSON-RPC errors."""
    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC Error {code}: {message}")


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def make_request(method: str, params: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 request object."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id or str(uuid.uuid4())
    }


def make_response(result: Any, request_id: str) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id
    }


def make_error_response(code: int, message: str, request_id: Optional[str] = None, data: Optional[Any] = None) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "error": error,
        "id": request_id
    }


def validate_request(request: Dict[str, Any]) -> bool:
    """Validate that a request conforms to JSON-RPC 2.0 spec."""
    if not isinstance(request, dict):
        return False
    if request.get("jsonrpc") != "2.0":
        return False
    if "method" not in request or not isinstance(request["method"], str):
        return False
    if "id" not in request:
        return False
    return True


# ─── Agent Card ──────────────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    """Schema for a tool's input/output."""
    input: Dict[str, str] = field(default_factory=dict)
    output: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentCard:
    """A2A Agent Card - describes agent capabilities for discovery."""
    identity: str
    capabilities: List[str]
    schemas: Dict[str, ToolSchema]
    endpoint: str
    version: str = "1.0"
    auth_methods: List[str] = field(default_factory=lambda: ["none"])
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "identity": self.identity,
            "capabilities": self.capabilities,
            "schemas": {
                name: {"input": schema.input, "output": schema.output}
                for name, schema in self.schemas.items()
            },
            "endpoint": self.endpoint,
            "version": self.version,
            "auth_methods": self.auth_methods,
        }
        if self.description:
            result["description"] = self.description
        return result


# ─── A2A Client ──────────────────────────────────────────────────────────────

class A2AClient:
    """Client for making A2A (Agent-to-Agent) JSON-RPC calls."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the A2A client.
        
        Args:
            base_url: Base URL of the target agent (e.g., "http://localhost:8002")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Fetch the agent's card from /.well-known/agent.json."""
        url = f"{self.base_url}/.well-known/agent.json"
        try:
            response = self._client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            log_to_loki("a2a.client", f"Failed to fetch agent card from {url}: {e}")
            raise JSONRPCError(INTERNAL_ERROR, f"Failed to fetch agent card: {e}")
    
    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a JSON-RPC call to the agent.
        
        Args:
            method: The JSON-RPC method to call
            params: Optional parameters for the method
            
        Returns:
            The result from the agent
            
        Raises:
            JSONRPCError: If the agent returns an error
        """
        url = f"{self.base_url}/api"
        request_id = str(uuid.uuid4())
        request_body = make_request(method, params, request_id)
        
        log_to_loki("a2a.client", f"Calling {method} on {self.base_url} with id={request_id}")
        
        try:
            response = self._client.post(
                url,
                json=request_body,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            # Check for JSON-RPC error
            if "error" in result:
                error = result["error"]
                log_to_loki("a2a.client", f"RPC error from {method}: {error}")
                raise JSONRPCError(
                    error.get("code", INTERNAL_ERROR),
                    error.get("message", "Unknown error"),
                    error.get("data")
                )
            
            log_to_loki("a2a.client", f"Received result from {method}")
            return result.get("result")
            
        except httpx.HTTPError as e:
            log_to_loki("a2a.client", f"HTTP error calling {method}: {e}")
            raise JSONRPCError(INTERNAL_ERROR, f"HTTP error: {e}")
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncA2AClient:
    """Async client for making A2A (Agent-to-Agent) JSON-RPC calls."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the async A2A client.
        
        Args:
            base_url: Base URL of the target agent (e.g., "http://localhost:8002")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Fetch the agent's card from /.well-known/agent.json."""
        url = f"{self.base_url}/.well-known/agent.json"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise JSONRPCError(INTERNAL_ERROR, f"Failed to fetch agent card: {e}")
    
    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make an async JSON-RPC call to the agent.
        
        Args:
            method: The JSON-RPC method to call
            params: Optional parameters for the method
            
        Returns:
            The result from the agent
            
        Raises:
            JSONRPCError: If the agent returns an error
        """
        url = f"{self.base_url}/api"
        request_id = str(uuid.uuid4())
        request_body = make_request(method, params, request_id)
        
        try:
            response = await self._client.post(
                url,
                json=request_body,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                error = result["error"]
                raise JSONRPCError(
                    error.get("code", INTERNAL_ERROR),
                    error.get("message", "Unknown error"),
                    error.get("data")
                )
            
            return result.get("result")
            
        except httpx.HTTPError as e:
            raise JSONRPCError(INTERNAL_ERROR, f"HTTP error: {e}")
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ─── Utilities ───────────────────────────────────────────────────────────────

def extract_method_and_params(request: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    """
    Extract method, params, and id from a validated JSON-RPC request.
    
    Returns:
        Tuple of (method, params, request_id)
    """
    return (
        request["method"],
        request.get("params", {}),
        request.get("id")
    )

