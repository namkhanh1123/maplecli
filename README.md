# OpenAI CLI

Multi-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Chat
openaicli chat
openaicli chat --model gpt-4 --temperature 0.9

# Images
openaicli image "sunset over mountains"
openaicli image "logo design" --model dall-e-3 --size 1792x1024

# Video
openaicli video "ocean waves"
openaicli video "fireworks display" --model sora-2

# Text-to-Speech
openaicli tts "Hello world"
openaicli tts "Story time" --voice nova --output story.mp3

# List models
openaicli models
openaicli models --type chat
openaicli models --type image
```

## Config

First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`

Or set environment variables:
```bash
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-your-key"
```

## Chat Commands

- `/save <file>` - Save chat
- `/load <file>` - Load chat
- `/clear` - Clear history
- `/temp 0.8` - Set temperature
- `/tokens 1000` - Set max tokens
- `/exit` - Quit

## Shortcuts

Add to PowerShell `$PROFILE`:

```powershell
function gemini {
    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
    $env:OPENAI_API_KEY = "your-key"
    openaicli @args
}
```

Use: `gemini chat --model gemini-2.5-pro`

See `examples/` for more shortcuts (Qwen, LM Studio, Ollama, etc.)
