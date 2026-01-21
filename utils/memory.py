from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single conversation turn (question + response)."""

    question: str
    plan: str
    response: str
    chart_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationMemory:
    """
    Manages conversation history for context retention.
    Keeps track of the last N conversation turns.
    """

    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to retain (default: 5)
        """
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.current_file_name: Optional[str] = None

    def add_turn(
        self, question: str, plan: str, response: str, chart_type: Optional[str] = None
    ) -> None:
        """
        Add a new conversation turn to memory.

        Args:
            question: User's question
            plan: Execution plan from Planner agent
            response: Final response from Executor agent
            chart_type: Type of chart generated (if any)
        """
        turn = ConversationTurn(
            question=question, plan=plan, response=response, chart_type=chart_type
        )
        self.turns.append(turn)

        # Keep only the last max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_context(self) -> str:
        """
        Get formatted conversation context for the LLM.

        Returns:
            Formatted string with recent conversation history
        """
        if not self.turns:
            return "No previous conversation."

        context_parts = ["Previous conversation:"]

        for i, turn in enumerate(self.turns, 1):
            context_parts.append(f"\n[Turn {i}]")
            context_parts.append(f"User: {turn.question}")
            context_parts.append(f"Response: {turn.response[:200]}...")
            if turn.chart_type:
                context_parts.append(f"(Generated: {turn.chart_type} chart)")

        return "\n".join(context_parts)

    def get_last_response(self) -> Optional[str]:
        """Get the last response for follow-up context."""
        if self.turns:
            return self.turns[-1].response
        return None

    def get_last_question(self) -> Optional[str]:
        """Get the last question for follow-up context."""
        if self.turns:
            return self.turns[-1].question
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns = []
        self.current_file_name = None

    def set_file(self, file_name: str) -> None:
        """
        Set current file and optionally clear history if file changed.

        Args:
            file_name: Name of the uploaded file
        """
        if self.current_file_name != file_name:
            self.clear()
            self.current_file_name = file_name

    def get_turns_count(self) -> int:
        """Get the number of stored conversation turns."""
        return len(self.turns)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation memory state.

        Returns:
            Dictionary with memory state information
        """
        return {
            "turns_count": len(self.turns),
            "max_turns": self.max_turns,
            "current_file": self.current_file_name,
            "has_context": len(self.turns) > 0,
        }
