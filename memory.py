"""
Simple in-memory conversation store keyed by session_id.
Not production-grade (would use Redis / Postgres in real life),
but good enough for demos and local testing.
"""

from collections import defaultdict, deque
from typing import Deque, List, Dict, Any
from app.config import get_settings
import uuid


class ConversationMemory:
    def __init__(self):
        self._store: Dict[str, Deque[Dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=get_settings().max_history_turns * 2)
        )

    def get_or_create_session(self, session_id: str | None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        # just touching the key creates the deque
        _ = self._store[session_id]
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._store[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._store[session_id])

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id].clear()


# singleton for the process
memory = ConversationMemory()
