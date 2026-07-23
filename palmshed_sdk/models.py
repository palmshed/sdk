"""Model configuration for Gemini AI SDK."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configurable model names. Override defaults via constructor kwargs."""

    text: str = "gemini-2.5-flash"
    thinking: str = "gemini-2.5-flash"
    url_context: str = "gemini-2.5-flash"
    image: str = "gemini-2.5-flash-image"
    deep_research: str = "deep-research-pro-preview-12-2025"
