"""Session management for Chainlit conversations.

Manages per-user session state, including:
- LangGraph agent instance per conversation thread
- Teaching progress tracking
- Session isolation between users
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Per-user session state."""
    user_id: str
    thread_id: str
    active_document: Optional[str] = None     # Currently loaded document
    total_questions: int = 0
    topics_covered: list[str] = field(default_factory=list)


class SessionManager:
    """Manages user sessions for the teaching assistant.

    In MVP mode (no auth), each Chainlit session gets its own
    UserSession. In production with auth, sessions are keyed by user ID.
    """

    def __init__(self):
        self._sessions: dict[str, UserSession] = {}
        self._counter = 0

    def get_or_create_session(self, user_id: Optional[str] = None) -> UserSession:
        """Get existing session or create a new one.

        Args:
            user_id: Unique user identifier. If None, generates an ID
                     and caches the session so repeated calls reuse it.

        Returns:
            UserSession for this user.
        """
        # Normalize: always have a cacheable key
        cache_key = user_id or "__anonymous__"

        if cache_key in self._sessions:
            return self._sessions[cache_key]

        self._counter += 1
        thread_id = f"thread_{self._counter}_{cache_key}"

        session = UserSession(
            user_id=user_id or cache_key,
            thread_id=thread_id,
        )

        self._sessions[cache_key] = session

        logger.info("Created session: user=%s, thread=%s", session.user_id, thread_id)
        return session

    def record_question(self, user_id: str, topic: str) -> None:
        """Record that a student asked about a topic."""
        if user_id in self._sessions:
            session = self._sessions[user_id]
            session.total_questions += 1
            if topic not in session.topics_covered:
                session.topics_covered.append(topic)

    def set_active_document(self, user_id: str, document_name: str) -> None:
        """Set the currently active document for a session."""
        if user_id in self._sessions:
            self._sessions[user_id].active_document = document_name

    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get session by user ID."""
        return self._sessions.get(user_id)


# Global singleton
session_manager = SessionManager()
