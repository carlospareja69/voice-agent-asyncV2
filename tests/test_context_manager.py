"""
Tests for agent/context/manager.py

These tests exercise the existing ContextManager behavior.  Role validation
is deferred to a future hardening pass (not required for pipeline correctness
in Issue #8 — the pipeline only calls add_turn with "user" or "assistant").
"""

from agent.context.manager import ContextManager


class TestInit:
    def test_default_system_prompt(self):
        """Default system prompt is present and appears in get_messages()."""
        cm = ContextManager()
        messages = cm.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"]  # non-empty

    def test_custom_system_prompt(self):
        """A custom system prompt replaces the default."""
        cm = ContextManager(system_prompt="You are a doctor.")
        messages = cm.get_messages()
        assert messages[0]["content"] == "You are a doctor."

    def test_initial_history_is_empty(self):
        """No turns are stored before add_turn() is called."""
        cm = ContextManager()
        assert len(cm.get_messages()) == 1  # system prompt only


class TestAddTurn:
    def test_user_turn_appended(self):
        """A user turn is appended after the system prompt."""
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        messages = cm.get_messages()
        assert len(messages) == 2
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_assistant_turn_appended(self):
        """An assistant turn is appended correctly."""
        cm = ContextManager()
        cm.add_turn("assistant", "Hi there")
        messages = cm.get_messages()
        assert messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_multiple_turns_preserved_in_order(self):
        """Turns are stored and returned in insertion order."""
        cm = ContextManager()
        cm.add_turn("user", "Q1")
        cm.add_turn("assistant", "A1")
        cm.add_turn("user", "Q2")
        messages = cm.get_messages()
        assert messages[1] == {"role": "user", "content": "Q1"}
        assert messages[2] == {"role": "assistant", "content": "A1"}
        assert messages[3] == {"role": "user", "content": "Q2"}

    def test_system_prompt_always_first(self):
        """The system prompt is always the first item regardless of turns added."""
        cm = ContextManager(system_prompt="custom prompt")
        cm.add_turn("user", "hello")
        cm.add_turn("assistant", "hi")
        messages = cm.get_messages()
        assert messages[0] == {"role": "system", "content": "custom prompt"}

    def test_get_messages_does_not_modify_internal_state(self):
        """Mutating the returned list does not affect the internal message store."""
        cm = ContextManager()
        cm.add_turn("user", "test")
        messages = cm.get_messages()
        messages.append({"role": "user", "content": "injected"})
        # Internal state unchanged
        assert len(cm.get_messages()) == 2  # system + 1 turn


class TestClear:
    def test_clear_removes_all_turns(self):
        """clear() removes all conversation turns."""
        cm = ContextManager()
        cm.add_turn("user", "hello")
        cm.add_turn("assistant", "hi")
        cm.clear()
        assert len(cm.get_messages()) == 1  # only system prompt

    def test_clear_preserves_system_prompt(self):
        """clear() keeps the system prompt intact."""
        cm = ContextManager(system_prompt="custom")
        cm.add_turn("user", "test")
        cm.clear()
        messages = cm.get_messages()
        assert messages[0]["content"] == "custom"

    def test_add_turn_works_after_clear(self):
        """Turns added after clear() are stored correctly."""
        cm = ContextManager()
        cm.add_turn("user", "first")
        cm.clear()
        cm.add_turn("user", "second")
        messages = cm.get_messages()
        assert len(messages) == 2
        assert messages[1]["content"] == "second"
