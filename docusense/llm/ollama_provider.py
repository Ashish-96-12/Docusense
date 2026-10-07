"""Local models through Ollama (https://ollama.com). Free and fully offline."""
from typing import List

import httpx

from docusense.llm.base import LLMError, LLMProvider
from docusense.schemas import ChatTurn


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def available(self) -> tuple[bool, str]:
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=1.5)
            resp.raise_for_status()
        except Exception:  # noqa: BLE001
            return False, f"Ollama not reachable at {self.host}"
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return False, f"model '{self.model}' not pulled (run: ollama pull {self.model})"
        return True, f"running at {self.host}"

    def complete(self, system: str, messages: List[ChatTurn], *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=180)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return resp.json().get("message", {}).get("content", "").strip()
