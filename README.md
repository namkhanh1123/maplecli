# MapleCLI<<<<<<< HEAD

# OpenAI CLI

Multi-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

Multi-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

## Install

## Install

```bash

pip install -e .```bash

```pip install -e .

```

## Usage

## Usage

```bash

# Chat```bash

openaicli chat# Chat

openaicli chat --model gpt-4 --temperature 0.9openaicli chat

openaicli chat --model gpt-4 --temperature 0.9

# Images

openaicli image "sunset over mountains"# Images

openaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli image "sunset over mountains"

openaicli image "logo design" --model dall-e-3 --size 1792x1024

# Video

openaicli video "ocean waves"# Video

openaicli video "fireworks display" --model sora-2openaicli video "ocean waves"

openaicli video "fireworks display" --model sora-2

# Text-to-Speech

openaicli tts "Hello world"# Text-to-Speech

openaicli tts "Story time" --voice nova --output story.mp3openaicli tts "Hello world"

openaicli tts "Story time" --voice nova --output story.mp3

# List models

openaicli models# List models

openaicli models --type chatopenaicli models

openaicli models --type imageopenaicli models --type chat

```openaicli models --type image

```

## Config

## Config

First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`

First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`

Or set environment variables:

```bashOr set environment variables:

export OPENAI_API_BASE="https://api.openai.com/v1"```bash

export OPENAI_API_KEY="sk-your-key"export OPENAI_API_BASE="https://api.openai.com/v1"

```export OPENAI_API_KEY="sk-your-key"

```

## Chat Commands

## Chat Commands

- `/save <file>` - Save chat

- `/load <file>` - Load chat- `/save <file>` - Save chat

- `/clear` - Clear history- `/load <file>` - Load chat

- `/temp 0.8` - Set temperature- `/clear` - Clear history

- `/tokens 1000` - Set max tokens- `/temp 0.8` - Set temperature

- `/exit` - Quit- `/tokens 1000` - Set max tokens

- `/exit` - Quit

## Shortcuts

## Shortcuts

Add to PowerShell `$PROFILE`:

Add to PowerShell `$PROFILE`:

```powershell

function maple {```powershell

    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"function gemini {

    $env:OPENAI_API_KEY = "your-key"    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

    openaicli @args    $env:OPENAI_API_KEY = "your-key"

}    openaicli @args

```}

```

Use: `maple chat --model deepseek-v3.2-exp`

Use: `gemini chat --model gemini-2.5-pro`

See `examples/` for more shortcuts (Qwen, LM Studio, Ollama, etc.)
=======
# maplecli
>>>>>>> c930605e9b544772195470cfa8defff53806b1e4
