"""Palmshed SDK - Gemini AI client for Python."""

__version__ = "0.1.0"

from .sdk import GeminiAI
from .models import ModelConfig
from .image_models import ImageConfig, ImageResult, ImageStatus

__all__ = ["GeminiAI", "ModelConfig", "ImageConfig", "ImageResult", "ImageStatus"]
