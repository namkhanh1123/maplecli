# MapleCLI# MapleCLI# MapleCLI<<<<<<< HEAD



Multi-feature CLI for OpenAI-compatible APIs with code analysis superpowers! 🚀



## FeaturesMulti-feature CLI for OpenAI-compatible APIs with Qwen-inspired interactive commands.# OpenAI CLI



✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  

🎨 **Rich Terminal UI**: Beautiful console output with Rich library  

⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  ## FeaturesMulti-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  

💾 **Session Management**: Save/load conversations  

🎲 **Random Seed Control**: Reproducible outputs  

🚀 **YOLO Mode**: Analyze entire codebases and get AI-powered insights!  ✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  Multi-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

📁 **Project Switching**: Navigate between different projects seamlessly

🎨 **Rich Terminal UI**: Beautiful console output with Rich library  

## Install

⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  ## Install

```bash

pip install -e .🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  

```

💾 **Session Management**: Save/load conversations  ## Install

## Usage

🎲 **Random Seed Control**: Reproducible outputs  

```bash

# Chat```bash

openaicli chat

openaicli chat --model gpt-4 --temperature 0.9## Install



# Imagespip install -e .```bash

openaicli image "sunset over mountains"

openaicli image "logo design" --model dall-e-3 --size 1792x1024```bash



# Videopip install -e .```pip install -e .

openaicli video "ocean waves"

openaicli video "fireworks display" --model sora-2```



# Text-to-Speech```

openaicli tts "Hello world"

openaicli tts "Story time" --voice nova --output story.mp3## Usage



# List models## Usage

openaicli models

openaicli models --type chat```bash

openaicli models --type image

```# Chat## Usage



## Interactive Chat Commandsopenaicli chat



When in chat mode, use these commands (with `:` or `/` prefix):openaicli chat --model gpt-4 --temperature 0.9```bash



### Essential Commands

- `:help` or `:h` - Show help message

- `:exit` or `:q` - Exit chat session# Images# Chat```bash

- `:clear` or `:cl` - Clear screen

- `:clear-history` or `:clh` - Clear conversation historyopenaicli image "sunset over mountains"



### Session Managementopenaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli chat# Chat

- `:save <file>` - Save conversation to file

- `:load <file>` - Load conversation from file

- `:history` or `:his` - Show conversation history

# Videoopenaicli chat --model gpt-4 --temperature 0.9openaicli chat

### Configuration

- `:conf` - Show current configurationopenaicli video "ocean waves"

- `:conf temperature=0.8` - Set temperature

- `:conf max_tokens=1000` - Set max tokensopenaicli video "fireworks display" --model sora-2openaicli chat --model gpt-4 --temperature 0.9

- `:reset-conf` - Reset to defaults



### YOLO Mode (Code Analysis) 🚀

- `:yolo` - Toggle YOLO mode on/off# Text-to-Speech# Images

- `:analyze` - Scan and analyze current project

- `:project` - Show current project path and recent projectsopenaicli tts "Hello world"

- `:project <path>` - Switch to a different project

- `:cd <path>` - Change directory (alias for :project)openaicli tts "Story time" --voice nova --output story.mp3openaicli image "sunset over mountains"# Images



**Example YOLO workflow:**

```

:yolo                           # Enable YOLO mode# List modelsopenaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli image "sunset over mountains"

:project ~/myapp                # Switch to your project

:analyze                        # Scan the codebaseopenaicli models

# Now ask questions about your code!

"What does the authentication system do?"openaicli models --type chatopenaicli image "logo design" --model dall-e-3 --size 1792x1024

"Find all API endpoints in this project"

"Explain the database schema"openaicli models --type image

```

```# Video

### Advanced

- `:seed` - Show current random seed

- `:seed 42` - Set random seed for reproducibility

- `:model` - Show current model## Interactive Chat Commandsopenaicli video "ocean waves"# Video



## Configuration



First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`When in chat mode, use these commands (with `:` or `/` prefix):openaicli video "fireworks display" --model sora-2openaicli video "ocean waves"



Or set environment variables:

```bash

export OPENAI_API_BASE="https://api.openai.com/v1"### Essential Commandsopenaicli video "fireworks display" --model sora-2

export OPENAI_API_KEY="sk-your-key"

```- `:help` or `:h` - Show help message



## Shortcuts- `:exit` or `:q` - Exit chat session# Text-to-Speech



Add to PowerShell `$PROFILE`:- `:clear` or `:cl` - Clear screen



```powershell- `:clear-history` or `:clh` - Clear conversation historyopenaicli tts "Hello world"# Text-to-Speech

function maple {

    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"

    $env:OPENAI_API_KEY = "your-key"

    openaicli @args### Session Managementopenaicli tts "Story time" --voice nova --output story.mp3openaicli tts "Hello world"

}

- `:save <file>` - Save conversation to file

function gemini {

    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"- `:load <file>` - Load conversation from fileopenaicli tts "Story time" --voice nova --output story.mp3

    $env:OPENAI_API_KEY = "your-key"

    openaicli @args- `:history` or `:his` - Show conversation history

}

```# List models



Usage:### Configuration

```powershell

maple chat --model deepseek-v3.2-exp- `:conf` - Show current configurationopenaicli models# List models

gemini chat --model gemini-2.5-pro

```- `:conf temperature=0.8` - Set temperature



## What's New- `:conf max_tokens=1000` - Set max tokensopenaicli models --type chatopenaicli models



### YOLO Mode Features- `:reset-conf` - Reset to defaults

- **Code Analysis**: Scan entire projects and understand structure

- **Project Switching**: Jump between different codebasesopenaicli models --type imageopenaicli models --type chat

- **Context Injection**: AI automatically gets project context

- **Smart Filtering**: Ignores node_modules, .git, venv, etc.### Advanced

- **File Statistics**: See lines of code by language

- **Tree Visualization**: Beautiful project structure display- `:seed` - Show current random seed```openaicli models --type image



### Qwen-Inspired Features- `:seed 42` - Set random seed for reproducibility

- **Colon Commands**: Use `:help` instead of `/help`

- **Screen Clearing**: `:clear` clears screen and shows welcome- `:model` - Show current model```

- **Random Seed**: `:seed` for reproducible outputs

- **Runtime Config**: `:conf key=value` without restarting

- **Organized Output**: Generated files in `generated/` folders

### Legacy Commands (still supported)## Config

## Input Tips

- `:temp 0.8` - Set temperature

- Type normally for single-line input

- End line with `\` for multiline input- `:tokens 1000` - Set max tokens## Config

- Press `Ctrl+D` to enter multiline mode

- Press `Ctrl+C` to interrupt (won't exit)



## Examples## ConfigurationFirst run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`



```bash

# Start chat with system prompt

openaicli chat --system "You are a helpful coding assistant"First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`



# Generate image with specific settings

openaicli image "cyberpunk city" --model dall-e-3 --quality hd --size 1792x1024

Or set environment variables:Or set environment variables:

# Generate video

openaicli video "time-lapse of sunset over ocean"```bash



# Text-to-speech with custom voiceexport OPENAI_API_BASE="https://api.openai.com/v1"```bashOr set environment variables:

openaicli tts "Welcome to MapleCLI" --voice nova --speed 1.2

export OPENAI_API_KEY="sk-your-key"

# YOLO mode for code review

openaicli chat```export OPENAI_API_BASE="https://api.openai.com/v1"```bash

:yolo

:analyze

# Ask: "Review this codebase for security issues"

```## Shortcutsexport OPENAI_API_KEY="sk-your-key"export OPENAI_API_BASE="https://api.openai.com/v1"



## Generated Files



All generated content is organized:Add to PowerShell `$PROFILE`:```export OPENAI_API_KEY="sk-your-key"

- Images → `generated/images/`

- Videos → `generated/videos/`

- Audio → `generated/audio/`

```powershell```

## Language Preference

function maple {

By default, responses are in English. To change:

```bash    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"## Chat Commands

openaicli chat --lang zh  # Chinese

openaicli chat --lang ja  # Japanese    $env:OPENAI_API_KEY = "your-key"

openaicli chat --lang es  # Spanish

```    openaicli @args## Chat Commands


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
