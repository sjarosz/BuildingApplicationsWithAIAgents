#!/usr/bin/env python3
"""
test_a2a_system.py
Test script for the A2A Orchestrator-Planner system.

Usage:
1. Start the Planner Agent (in terminal 1):
   python -m src.common.a2a.planner_agent

2. Start the Orchestrator Agent (in terminal 2):
   python -m src.common.a2a.orchestrator_agent

3. Run this test script (in terminal 3):
   python -m src.common.a2a.test_a2a_system
"""

import json
import httpx
from typing import Any, Dict

ORCHESTRATOR_URL = "http://localhost:8001"
PLANNER_URL = "http://localhost:8002"


def make_rpc_request(url: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a JSON-RPC request to an agent."""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    
    response = httpx.post(
        f"{url}/api",
        json=request,
        headers={"Content-Type": "application/json"},
        timeout=30.0
    )
    
    return response.json()


def test_agent_cards():
    """Test that agent cards are accessible and A2A compliant."""
    print("=" * 60)
    print("Testing Agent Cards (A2A Spec)")
    print("=" * 60)
    
    # Test Orchestrator agent card
    try:
        response = httpx.get(f"{ORCHESTRATOR_URL}/.well-known/agent.json")
        card = response.json()
        print(f"\n✓ Orchestrator Agent Card:")
        print(f"  Name: {card.get('name', card.get('identity', 'N/A'))}")
        print(f"  URL: {card.get('url', card.get('endpoint', 'N/A'))}")
        print(f"  Version: {card.get('version', 'N/A')}")
        print(f"  Protocol: {card.get('protocol', 'N/A')}")
        if 'skills' in card:
            print(f"  Skills: {[s['id'] for s in card['skills']]}")
        elif 'capabilities' in card:
            print(f"  Capabilities: {card['capabilities']}")
        print(f"  Description: {card.get('description', 'N/A')[:80]}...")
    except Exception as e:
        print(f"\n✗ Orchestrator Agent not available: {e}")
        return False
    
    # Test Planner agent card
    try:
        response = httpx.get(f"{PLANNER_URL}/.well-known/agent.json")
        card = response.json()
        print(f"\n✓ Planner Agent Card:")
        print(f"  Name: {card.get('name', card.get('identity', 'N/A'))}")
        print(f"  URL: {card.get('url', card.get('endpoint', 'N/A'))}")
        print(f"  Version: {card.get('version', 'N/A')}")
        print(f"  Protocol: {card.get('protocol', 'N/A')}")
        if 'skills' in card:
            print(f"  Skills: {[s['id'] for s in card['skills']]}")
        elif 'capabilities' in card:
            print(f"  Capabilities: {card['capabilities']}")
        if 'supportedTools' in card:
            print(f"  Supported Tools: {[t['name'] for t in card['supportedTools']]}")
        print(f"  Description: {card.get('description', 'N/A')[:80]}...")
    except Exception as e:
        print(f"\n✗ Planner Agent not available: {e}")
        return False
    
    return True


def test_list_tools():
    """Test the listTools method."""
    print("\n" + "=" * 60)
    print("Testing listTools")
    print("=" * 60)
    
    result = make_rpc_request(ORCHESTRATOR_URL, "listTools", {})
    
    if "result" in result:
        tools = result["result"]["tools"]
        print(f"\n✓ Available tools: {tools}")
        return True
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        return False


def test_simple_calculation():
    """Test a simple calculation task."""
    print("\n" + "=" * 60)
    print("Testing Simple Calculation")
    print("=" * 60)
    
    task = "What is 15 multiplied by 7?"
    print(f"\nTask: {task}")
    
    result = make_rpc_request(ORCHESTRATOR_URL, "executeTask", {"task": task})
    
    if "result" in result:
        print(f"\n✓ Success!")
        print(f"  Summary: {result['result']['summary']}")
        print(f"  Tool Results: {json.dumps(result['result']['task_results'], indent=2)}")
        return True
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        return False


def test_multiple_tools():
    """Test a task requiring multiple tools."""
    print("\n" + "=" * 60)
    print("Testing Multiple Tools")
    print("=" * 60)
    
    task = "Calculate 100 divided by 4, then tell me the current time"
    print(f"\nTask: {task}")
    
    result = make_rpc_request(ORCHESTRATOR_URL, "executeTask", {"task": task})
    
    if "result" in result:
        print(f"\n✓ Success!")
        print(f"  Summary: {result['result']['summary']}")
        print(f"  Planning Reasoning: {result['result'].get('planning_reasoning', 'N/A')}")
        print(f"  Tool Results:")
        for r in result['result']['task_results']:
            status = "✓" if r['success'] else "✗"
            print(f"    {status} {r['tool']}: {r['result']}")
        return True
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        return False


def test_text_formatting():
    """Test text formatting tools."""
    print("\n" + "=" * 60)
    print("Testing Text Formatting")
    print("=" * 60)
    
    task = "Convert 'hello world' to uppercase and also reverse it"
    print(f"\nTask: {task}")
    
    result = make_rpc_request(ORCHESTRATOR_URL, "executeTask", {"task": task})
    
    if "result" in result:
        print(f"\n✓ Success!")
        print(f"  Summary: {result['result']['summary']}")
        print(f"  Tool Results:")
        for r in result['result']['task_results']:
            status = "✓" if r['success'] else "✗"
            print(f"    {status} {r['tool']}: {r['result']}")
        return True
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        return False


def test_chain_calculation():
    """Test a chained calculation."""
    print("\n" + "=" * 60)
    print("Testing Chain Calculation")
    print("=" * 60)
    
    task = "Add 50 and 25, then multiply 10 by 3"
    print(f"\nTask: {task}")
    
    result = make_rpc_request(ORCHESTRATOR_URL, "executeTask", {"task": task})
    
    if "result" in result:
        print(f"\n✓ Success!")
        print(f"  Summary: {result['result']['summary']}")
        print(f"  Tool Results:")
        for r in result['result']['task_results']:
            status = "✓" if r['success'] else "✗"
            print(f"    {status} {r['tool']}({r['params']}): {r['result']}")
        return True
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("A2A ORCHESTRATOR-PLANNER SYSTEM TEST")
    print("=" * 60)
    
    # Check if agents are running
    if not test_agent_cards():
        print("\n" + "!" * 60)
        print("AGENTS NOT RUNNING!")
        print("Please start the agents first:")
        print("  Terminal 1: python -m src.common.a2a.planner_agent")
        print("  Terminal 2: python -m src.common.a2a.orchestrator_agent")
        print("!" * 60)
        return
    
    # Run tests
    tests = [
        ("List Tools", test_list_tools),
        ("Simple Calculation", test_simple_calculation),
        ("Multiple Tools", test_multiple_tools),
        ("Text Formatting", test_text_formatting),
        ("Chain Calculation", test_chain_calculation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()

