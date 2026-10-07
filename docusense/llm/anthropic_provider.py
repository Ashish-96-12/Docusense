from typing import List, Optional

from docusense.llm.base import LLMError, LLMProvider
from docusense.schemas import ChatTurn


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str], model: str):
        self.api_key = api_key
        self.model = model
        self._client = None

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, "API key set"

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMError("ANTHROPIC_API_KEY not set")
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key, max_retries=2, timeout=60)
        return self._client

    def complete(self, system: str, messages: List[ChatTurn], *, temperature: float, max_tokens: int) -> str:
        request = {
            "model": self.model,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        client = self._get_client()
        try:
            try:
                resp = client.messages.create(**request)
            except Exception as exc:  # noqa: BLE001
                # Some models only accept their default sampling settings
                if "temperature" not in str(exc).lower():
                    raise
                request.pop("temperature")
                resp = client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - surface any SDK error the same way
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text").strip()
