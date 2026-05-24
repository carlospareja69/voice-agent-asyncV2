class ContextManager:
    """Maintains conversation history and the system prompt for the LLM."""

    def __init__(self, system_prompt: str = "You are a helpful voice assistant.") -> None:
        self.system_prompt = system_prompt
        self._messages: list[dict] = []

    def add_turn(self, role: str, content: str) -> None:
        """Append a single message turn. role must be 'user' or 'assistant'."""
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return the full message list, prefixed with the system prompt."""
        return [{"role": "system", "content": self.system_prompt}] + self._messages

    def clear(self) -> None:
        """Reset conversation history, keeping the system prompt."""
        self._messages = []
