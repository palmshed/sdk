# palmshed-sdk

Gemini AI client for Python. Text generation, chat, thinking, URL context, image generation, text-to-speech, and deep research.

## Install

```bash
pip install palmshed-sdk
```

With optional dependencies:

```bash
pip install palmshed-sdk[redis]    # Redis caching
pip install palmshed-sdk[vertex]   # Vertex AI image generation
```

## Usage

```python
from palmshed_sdk import GeminiAI

client = GeminiAI(api_key="your-key")

# Simple text generation
response = client.generate_text("Write a haiku about Python")

# Chat with history
messages = [
    {"role": "user", "content": "What is Flask?"},
    {"role": "assistant", "content": "Flask is a lightweight web framework."},
    {"role": "user", "content": "How do I install it?"},
]
response = client.generate_chat(messages)

# Thinking mode
result = client.generate_text_with_thinking("Explain quantum computing")
print(result["response"])
print(result["thinking_summary"])

# Text-to-speech
filepath = client.text_to_speech("Hello world")

# Image generation
result = client.generate_image("A sunset over the ocean")
print(result.filepath)
```

## Configuration

```python
from palmshed_sdk import GeminiAI, ModelConfig

# Custom model names
models = ModelConfig(
    text="gemini-2.5-flash",
    thinking="gemini-2.5-pro",
    image="gemini-2.5-flash-image",
)

client = GeminiAI(
    api_key="your-key",
    models=models,
    redis_url="redis://localhost:6379",
    go_service_url="http://localhost:8080/process",
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Required. Your Gemini API key. |
| `REDIS_URL` | Optional. Redis URL for caching. |
| `IMAGE_PROVIDER` | Optional. Image provider: `gemini`, `vertex`, `mock`. |
| `IMAGE_MODEL` | Optional. Image model name. |
| `GOOGLE_CLOUD_PROJECT` | Optional. GCP project for Vertex AI. |
| `GOOGLE_CLOUD_LOCATION` | Optional. GCP location for Vertex AI. |

## License

MIT
