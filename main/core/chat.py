# core/chat.py - Chat functionality for the DocuChat application
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None


class ChatHistory:
    """Manages chat history and conversation state"""

    def __init__(self, max_history: int = 50):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a new message to the chat history"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Keep only the last max_history messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_messages(self) -> List[ChatMessage]:
        """Get all messages in the chat history"""
        return self.messages.copy()

    def get_recent_context(self, num_messages: int = 5) -> str:
        """Get recent conversation context as a formatted string"""
        recent_messages = self.messages[-num_messages:] if len(self.messages) > num_messages else self.messages

        context_parts = []
        for msg in recent_messages:
            role_label = "Human" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")

        return "\n".join(context_parts)

    def clear_history(self):
        """Clear all chat history"""
        self.messages.clear()

    def export_history(self) -> List[Dict]:
        """Export chat history as a list of dictionaries"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in self.messages
        ]


class ChatManager:
    """Manages multiple chat sessions"""

    def __init__(self):
        self.sessions: Dict[str, ChatHistory] = {}

    def get_or_create_session(self, session_id: str) -> ChatHistory:
        """Get existing session or create a new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatHistory()
        return self.sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())
