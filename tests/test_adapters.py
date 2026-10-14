"""Offline checks of provider wire formats (no network)."""

from agentarena.agents.anthropic_agent import AnthropicModel
from agentarena.agents.openai_compat_agent import OpenAICompatModel
from agentarena.schemas import Message, ToolCall, ToolResult

HISTORY = [
    Message(role="system", content="sys"),
    Message(role="user", content="buy it"),
    Message(role="assistant", content="Looking.", tool_calls=[
        ToolCall(id="t1", name="search_products", arguments={"query": "x"}),
        ToolCall(id="t2", name="get_balance", arguments={}),
    ]),
    Message(role="tool", tool_results=[
        ToolResult(call_id="t1", name="search_products", content="[]"),
        ToolResult(call_id="t2", name="get_balance", content="ERROR: nope", is_error=True, error_kind="domain"),
    ]),
]


def test_anthropic_wire_groups_parallel_results_in_one_user_turn():
    system, wire = AnthropicModel.to_wire(HISTORY)
    assert system == "sys"
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert [b["type"] for b in wire[1]["content"]] == ["text", "tool_use", "tool_use"]
    results = wire[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and results[1]["is_error"] is True


def test_anthropic_replays_provider_raw_verbatim():
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]
    _, wire = AnthropicModel.to_wire([Message(role="user", content="q"),
                                      Message(role="assistant", tool_calls=[], provider_raw=raw)])
    assert wire[1]["content"] is raw


def test_openai_wire_one_tool_message_per_result():
    wire = OpenAICompatModel.to_wire(HISTORY)
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "tool"]
    assert wire[2]["tool_calls"][0]["function"] == {"name": "search_products", "arguments": '{"query": "x"}'}
    assert wire[4]["tool_call_id"] == "t2"


def test_openai_sdk_surface_exists():
    import openai

    client = openai.AsyncOpenAI(api_key="k", base_url="https://api.z.ai/api/paas/v4/")
    assert callable(client.chat.completions.create)
