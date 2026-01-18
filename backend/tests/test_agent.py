#!/usr/bin/env python3
"""
Test Script for AI Agent System
Verify that everything is working before starting the server
"""

import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")

    try:
        from app.agent.base.base_agent import BaseAgent, AgentState
        print("  ✅ BaseAgent imported")

        from app.agent.core.registry import AgentRegistry
        print("  ✅ AgentRegistry imported")

        from app.agent.tools.base_tool import BaseTool, ToolRegistry
        print("  ✅ Tool system imported")

        from app.agent.tools.service_tools import (
            check_service_status_tool,
            start_service_tool,
            stop_service_tool,
            get_service_logs_tool
        )
        print("  ✅ Service tools imported")

        from app.agent.agents.service_agent import ServiceManagementAgent
        print("  ✅ ServiceAgent imported")

        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_key():
    """Test that OpenAI API key is configured"""
    print("\n🔍 Testing OpenAI configuration...")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"  ✅ OPENAI_API_KEY found: {api_key[:10]}...")
        return True
    else:
        print("  ❌ OPENAI_API_KEY not found in environment")
        print("     Please set it in backend/.env file")
        return False


def test_agent_registry():
    """Test that agents are registered"""
    print("\n🔍 Testing agent registry...")

    try:
        from app.agent.core.registry import AgentRegistry

        agents = AgentRegistry.list_agents()

        if not agents:
            print("  ❌ No agents registered")
            return False

        print(f"  ✅ Found {len(agents)} registered agent(s):")
        for agent_info in agents:
            print(f"     - {agent_info['agent_id']}: {agent_info['class_name']}")

        return True
    except Exception as e:
        print(f"  ❌ Registry test failed: {e}")
        return False


def test_tool_registry():
    """Test that tools are registered"""
    print("\n🔍 Testing tool registry...")

    try:
        from app.agent.tools.base_tool import ToolRegistry

        tools = ToolRegistry.list_tools()

        if not tools:
            print("  ❌ No tools registered")
            return False

        print(f"  ✅ Found {len(tools)} registered tool(s):")
        for tool_info in tools:
            print(f"     - {tool_info['name']} ({tool_info['category']})")

        return True
    except Exception as e:
        print(f"  ❌ Tool registry test failed: {e}")
        return False


def test_agent_creation():
    """Test creating an agent instance"""
    print("\n🔍 Testing agent creation...")

    try:
        from app.agent.core.registry import AgentRegistry

        agent = AgentRegistry.get_agent("service_management")
        print(f"  ✅ Created agent: {agent.__class__.__name__}")
        print(f"     Model: {agent.model_name}")
        print(f"     Tools: {len(agent.tools)}")

        return True
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_execution():
    """Test executing a simple tool"""
    print("\n🔍 Testing tool execution...")

    try:
        from app.agent.tools.service_tools import check_service_status

        result = check_service_status("camera_management")
        print(f"  ✅ Tool executed successfully")
        print(f"     Status: {result.get('status')}")
        print(f"     Running: {result.get('is_running')}")
        print(f"     WebSocket: {result.get('websocket_connected')}")

        return True
    except Exception as e:
        print(f"  ❌ Tool execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("AI Agent System Test")
    print("=" * 80)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("OpenAI Configuration", test_openai_key()))
    results.append(("Agent Registry", test_agent_registry()))
    results.append(("Tool Registry", test_tool_registry()))
    results.append(("Agent Creation", test_agent_creation()))
    results.append(("Tool Execution", test_tool_execution()))

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests passed! Agent system is ready to use.")
        print("\nNext steps:")
        print("  1. Start backend server: cd backend && uvicorn app.main:app --reload")
        print("  2. Test agent API: http://localhost:8000/docs")
        print("  3. Try chatting: POST /api/agent/chat")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
