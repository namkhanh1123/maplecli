# MapleCLI# MapleCLI<<<<<<< HEAD



Multi-feature CLI for OpenAI-compatible APIs with Qwen-inspired interactive commands.# OpenAI CLI



## FeaturesMulti-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.



✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  Multi-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

🎨 **Rich Terminal UI**: Beautiful console output with Rich library  

⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  ## Install

🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  

💾 **Session Management**: Save/load conversations  ## Install

🎲 **Random Seed Control**: Reproducible outputs  

```bash

## Install

pip install -e .```bash

```bash

pip install -e .```pip install -e .

```

```

## Usage

## Usage

```bash

# Chat## Usage

openaicli chat

openaicli chat --model gpt-4 --temperature 0.9```bash



# Images# Chat```bash

openaicli image "sunset over mountains"

openaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli chat# Chat



# Videoopenaicli chat --model gpt-4 --temperature 0.9openaicli chat

openaicli video "ocean waves"

openaicli video "fireworks display" --model sora-2openaicli chat --model gpt-4 --temperature 0.9



# Text-to-Speech# Images

openaicli tts "Hello world"

openaicli tts "Story time" --voice nova --output story.mp3openaicli image "sunset over mountains"# Images



# List modelsopenaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli image "sunset over mountains"

openaicli models

openaicli models --type chatopenaicli image "logo design" --model dall-e-3 --size 1792x1024

openaicli models --type image

```# Video



## Interactive Chat Commandsopenaicli video "ocean waves"# Video



When in chat mode, use these commands (with `:` or `/` prefix):openaicli video "fireworks display" --model sora-2openaicli video "ocean waves"



### Essential Commandsopenaicli video "fireworks display" --model sora-2

- `:help` or `:h` - Show help message

- `:exit` or `:q` - Exit chat session# Text-to-Speech

- `:clear` or `:cl` - Clear screen

- `:clear-history` or `:clh` - Clear conversation historyopenaicli tts "Hello world"# Text-to-Speech



### Session Managementopenaicli tts "Story time" --voice nova --output story.mp3openaicli tts "Hello world"

- `:save <file>` - Save conversation to file

- `:load <file>` - Load conversation from fileopenaicli tts "Story time" --voice nova --output story.mp3

- `:history` or `:his` - Show conversation history

# List models

### Configuration

- `:conf` - Show current configurationopenaicli models# List models

- `:conf temperature=0.8` - Set temperature

- `:conf max_tokens=1000` - Set max tokensopenaicli models --type chatopenaicli models

- `:reset-conf` - Reset to defaults

openaicli models --type imageopenaicli models --type chat

### Advanced

- `:seed` - Show current random seed```openaicli models --type image

- `:seed 42` - Set random seed for reproducibility

- `:model` - Show current model```



### Legacy Commands (still supported)## Config

- `:temp 0.8` - Set temperature

- `:tokens 1000` - Set max tokens## Config



## ConfigurationFirst run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`



First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`



Or set environment variables:Or set environment variables:

```bash

export OPENAI_API_BASE="https://api.openai.com/v1"```bashOr set environment variables:

export OPENAI_API_KEY="sk-your-key"

```export OPENAI_API_BASE="https://api.openai.com/v1"```bash



## Shortcutsexport OPENAI_API_KEY="sk-your-key"export OPENAI_API_BASE="https://api.openai.com/v1"



Add to PowerShell `$PROFILE`:```export OPENAI_API_KEY="sk-your-key"



```powershell```

function maple {

    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"## Chat Commands

    $env:OPENAI_API_KEY = "your-key"

    openaicli @args## Chat Commands

}

- `/save <file>` - Save chat

function gemini {

    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"- `/load <file>` - Load chat- `/save <file>` - Save chat

    $env:OPENAI_API_KEY = "your-key"

    openaicli @args- `/clear` - Clear history- `/load <file>` - Load chat

}

```- `/temp 0.8` - Set temperature- `/clear` - Clear history



Usage:- `/tokens 1000` - Set max tokens- `/temp 0.8` - Set temperature

```powershell

maple chat --model deepseek-v3.2-exp- `/exit` - Quit- `/tokens 1000` - Set max tokens

gemini chat --model gemini-2.5-pro

```- `/exit` - Quit



## What's New (Qwen-Inspired Features)## Shortcuts



- **Colon Commands**: Use `:help` instead of `/help` (Qwen-style)## Shortcuts

- **Screen Clearing**: `:clear` clears screen and shows welcome message

- **Random Seed**: `:seed` for reproducible outputsAdd to PowerShell `$PROFILE`:

- **Runtime Config**: `:conf key=value` to change settings without restarting

- **Better Welcome**: Cleaner welcome screen with helpful hintsAdd to PowerShell `$PROFILE`:

- **Organized Output**: Generated files saved to `generated/` folders

```powershell

## Input Tips

function maple {```powershell

- Type normally for single-line input

- End line with `\` for multiline input    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"function gemini {

- Press `Ctrl+D` to enter multiline mode

- Press `Ctrl+C` to interrupt (won't exit)    $env:OPENAI_API_KEY = "your-key"    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"



## Examples    openaicli @args    $env:OPENAI_API_KEY = "your-key"



```bash}    openaicli @args

# Start chat with system prompt

openaicli chat --system "You are a helpful coding assistant"```}



# Generate image with specific settings```

openaicli image "cyberpunk city" --model dall-e-3 --quality hd --size 1792x1024

Use: `maple chat --model deepseek-v3.2-exp`

# Generate video

openaicli video "time-lapse of sunset over ocean"Use: `gemini chat --model gemini-2.5-pro`



# Text-to-speech with custom voiceSee `examples/` for more shortcuts (Qwen, LM Studio, Ollama, etc.)

openaicli tts "Welcome to MapleCLI" --voice nova --speed 1.2=======

```# maplecli

>>>>>>> c930605e9b544772195470cfa8defff53806b1e4

## Generated Files

All generated content is organized:
- Images → `generated/images/`
- Videos → `generated/videos/`
- Audio → `generated/audio/`
