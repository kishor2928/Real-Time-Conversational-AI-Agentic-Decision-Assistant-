import pytest
from app.agent.tools import calculate, get_current_time, lookup_exchange_rate
from app.agent.memory import ConversationMemory


def test_calculate_basic():
    result = calculate.invoke({"expression": "2 + 2"})
    assert result == "4"


def test_calculate_rejects_injection():
    result = calculate.invoke({"expression": "__import__('os').system('ls')"})
    assert "Error" in result or "error" in result.lower()


def test_get_current_time():
    result = get_current_time.invoke({})
    assert "UTC" in result
    assert len(result) > 10


def test_memory_session_creation():
    mem = ConversationMemory()
    sid = mem.get_or_create_session(None)
    assert sid is not None
    mem.add_message(sid, "user", "hello")
    hist = mem.get_history(sid)
    assert len(hist) == 1
    assert hist[0]["content"] == "hello"
