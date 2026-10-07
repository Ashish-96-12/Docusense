from typing import List, Optional

from docusense.llm.base import LLMError, LLMProvider
from docusense.schemas import ChatTurn


class OpenAIProvider(LLMProvider):
    """Works with OpenAI and any OpenAI-compatible server (set base_url)."""

    name = "openai"

    def __init__(self, api_key: Optional[str], model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OPENAI_API_KEY not set"
        return True, "API key set"

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMError("OPENAI_API_KEY not set")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=2, timeout=60)
        return self._client

    def complete(self, system: str, messages: List[ChatTurn], *, temperature: float, max_tokens: int) -> str:
        payload = [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model, messages=payload, temperature=temperature, max_completion_tokens=max_tokens
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()
