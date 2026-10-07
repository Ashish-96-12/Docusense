"""Pick an LLM provider by name, or automatically.

"auto" uses the first one that's usable: Claude, then OpenAI, then a local
Ollama model, then the offline extractive fallback.
"""
from typing import Dict

from docusense.config import Settings
from docusense.llm.anthropic_provider import AnthropicProvider
from docusense.llm.base import LLMProvider
from docusense.llm.extractive import ExtractiveProvider
from docusense.llm.ollama_provider import OllamaProvider
from docusense.llm.openai_provider import OpenAIProvider

AUTO_ORDER = ("anthropic", "openai", "ollama", "extractive")


def build_providers(settings: Settings) -> Dict[str, LLMProvider]:
    return {
        "anthropic": AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model),
        "openai": OpenAIProvider(settings.openai_api_key, settings.openai_model, settings.openai_base_url),
        "ollama": OllamaProvider(settings.ollama_host, settings.ollama_model),
        "extractive": ExtractiveProvider(),
    }


def resolve_provider(name: str, providers: Dict[str, LLMProvider]) -> LLMProvider:
    if name != "auto":
        if name not in providers:
            raise ValueError(f"Unknown provider '{name}'. Choose from: auto, {', '.join(providers)}")
        return providers[name]
    for candidate in AUTO_ORDER:
        ok, _ = providers[candidate].available()
        if ok:
            return providers[candidate]
    return providers["extractive"]
