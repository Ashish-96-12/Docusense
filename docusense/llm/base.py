"""Common interface every LLM provider implements."""
from abc import ABC, abstractmethod
from typing import List, Optional

from docusense.schemas import ChatTurn, RetrievedChunk


class LLMError(RuntimeError):
    """Raised when a provider call fails (bad key, network, rate limit...)."""


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(is_usable, human-readable reason)."""

    @abstractmethod
    def complete(self, system: str, messages: List[ChatTurn], *, temperature: float, max_tokens: int) -> str:
        """Return the assistant's reply text for a chat conversation."""

    def answer(
        self,
        question: str,
        context: List[RetrievedChunk],
        system: str,
        messages: List[ChatTurn],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Hook for providers that answer from context directly (the extractive one).

        LLM-backed providers just complete the prompt the pipeline already built.
        """
        return self.complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def require(value: Optional[str], what: str) -> str:
    if not value:
        raise LLMError(f"{what} is not set")
    return value
