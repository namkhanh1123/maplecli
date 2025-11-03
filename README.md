# MapleCLI# MapleCLI# MapleCLI# MapleCLI<<<<<<< HEAD



Multi-feature CLI for OpenAI-compatible APIs with code analysis superpowers! 🚀



## FeaturesMulti-feature CLI for OpenAI-compatible APIs with code analysis superpowers! 🚀



✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  

🎨 **Rich Terminal UI**: Beautiful console output with Rich library  

⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  ## FeaturesMulti-feature CLI for OpenAI-compatible APIs with Qwen-inspired interactive commands.# OpenAI CLI

🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  

💾 **Session Management**: Save/load conversations  

🎲 **Random Seed Control**: Reproducible outputs  

🚀 **YOLO Mode**: Analyze entire codebases and get AI-powered insights!  ✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  

📁 **Project Switching**: Navigate between different projects seamlessly

🎨 **Rich Terminal UI**: Beautiful console output with Rich library  

## Install

⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  ## FeaturesMulti-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

```bash

pip install -e .🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  

```

💾 **Session Management**: Save/load conversations  

## Usage

🎲 **Random Seed Control**: Reproducible outputs  

```bash

# Chat🚀 **YOLO Mode**: Analyze entire codebases and get AI-powered insights!  ✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  Multi-feature CLI for OpenAI-compatible APIs: chat, images, videos, text-to-speech.

openaicli chat

openaicli chat --model gpt-4 --temperature 0.9📁 **Project Switching**: Navigate between different projects seamlessly



# Images🎨 **Rich Terminal UI**: Beautiful console output with Rich library  

openaicli image "sunset over mountains"

openaicli image "logo design" --model dall-e-3 --size 1792x1024## Install



# Video⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  ## Install

openaicli video "ocean waves"

openaicli video "fireworks display" --model sora-2```bash



# Text-to-Speechpip install -e .🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  

openaicli tts "Hello world"

openaicli tts "Story time" --voice nova --output story.mp3```



# List models💾 **Session Management**: Save/load conversations  ## Install

openaicli models

openaicli models --type chat## Usage

openaicli models --type image

```🎲 **Random Seed Control**: Reproducible outputs  



## Interactive Chat Commands```bash



When in chat mode, use these commands (with `:` or `/` prefix):# Chat```bash



### Essential Commandsopenaicli chat

- `:help` or `:h` - Show help message

- `:exit` or `:q` - Exit chat sessionopenaicli chat --model gpt-4 --temperature 0.9## Install

- `:clear` or `:cl` - Clear screen

- `:clear-history` or `:clh` - Clear conversation history



### Session Management# Imagespip install -e .```bash

- `:save <file>` - Save conversation to file

- `:load <file>` - Load conversation from fileopenaicli image "sunset over mountains"

- `:history` or `:his` - Show conversation history

openaicli image "logo design" --model dall-e-3 --size 1792x1024```bash

### Configuration

- `:conf` - Show current configuration

- `:conf temperature=0.8` - Set temperature

- `:conf max_tokens=1000` - Set max tokens# Videopip install -e .```pip install -e .

- `:reset-conf` - Reset to defaults

openaicli video "ocean waves"

### YOLO Mode (Code Analysis) 🚀

- `:yolo` - Toggle YOLO mode on/offopenaicli video "fireworks display" --model sora-2```

- `:analyze` - Scan and analyze current project (auto-reads key files)

- `:read <file>` - Read a specific file and add to context

- `:files` - List all files in current project

- `:files <pattern>` - Filter files by pattern (e.g., `:files tsx`)# Text-to-Speech```

- `:project` - Show current project path and recent projects

- `:project <path>` - Switch to a different projectopenaicli tts "Hello world"

- `:cd <path>` - Change directory (alias for :project)

openaicli tts "Story time" --voice nova --output story.mp3## Usage

**Example YOLO workflow:**

```

:yolo                           # Enable YOLO mode

:cd C:\Projects\myapp           # Switch to your project# List models## Usage

:analyze                        # Scan codebase (auto-reads 10 key files)

:files api                      # List all API-related filesopenaicli models

:read src/services/auth.ts      # Read specific file

# Now AI has full context - ask anything!openaicli models --type chat```bash

"What does the authentication system do?"

"Find all API endpoints"openaicli models --type image

"Review for security issues"

"Explain the database schema"```# Chat## Usage

"How does user registration work?"

```



**What :analyze does:**## Interactive Chat Commandsopenaicli chat

- Scans entire project structure

- Counts lines by file type

- Auto-reads up to 10 important files (README, package.json, main files, etc.)

- Injects all code into AI contextWhen in chat mode, use these commands (with `:` or `/` prefix):openaicli chat --model gpt-4 --temperature 0.9```bash

- AI can now answer detailed questions about your codebase!



### Advanced

- `:seed` - Show current random seed### Essential Commands

- `:seed 42` - Set random seed for reproducibility

- `:model` - Show current model- `:help` or `:h` - Show help message



## Configuration- `:exit` or `:q` - Exit chat session# Images# Chat```bash



First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`- `:clear` or `:cl` - Clear screen



Or set environment variables:- `:clear-history` or `:clh` - Clear conversation historyopenaicli image "sunset over mountains"

```bash

export OPENAI_API_BASE="https://api.openai.com/v1"

export OPENAI_API_KEY="sk-your-key"

```### Session Managementopenaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli chat# Chat



## Shortcuts- `:save <file>` - Save conversation to file



Add to PowerShell `$PROFILE`:- `:load <file>` - Load conversation from file



```powershell- `:history` or `:his` - Show conversation history

function maple {

    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"# Videoopenaicli chat --model gpt-4 --temperature 0.9openaicli chat

    $env:OPENAI_API_KEY = "your-key"

    openaicli @args### Configuration

}

- `:conf` - Show current configurationopenaicli video "ocean waves"

function gemini {

    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"- `:conf temperature=0.8` - Set temperature

    $env:OPENAI_API_KEY = "your-key"

    openaicli @args- `:conf max_tokens=1000` - Set max tokensopenaicli video "fireworks display" --model sora-2openaicli chat --model gpt-4 --temperature 0.9

}

```- `:reset-conf` - Reset to defaults



Usage:

```powershell

maple chat --model deepseek-v3.2-exp### YOLO Mode (Code Analysis) 🚀

gemini chat --model gemini-2.5-pro

```- `:yolo` - Toggle YOLO mode on/off# Text-to-Speech# Images



## What's New- `:analyze` - Scan and analyze current project



### YOLO Mode Features ⚡- `:project` - Show current project path and recent projectsopenaicli tts "Hello world"

- **Full Code Analysis**: `:analyze` now reads actual file contents (not just structure)

- **Smart File Selection**: Auto-picks important files (README, package.json, main.*, index.*, etc.)- `:project <path>` - Switch to a different project

- **On-Demand Reading**: `:read <file>` to add specific files to context

- **File Discovery**: `:files` to list all files, `:files <pattern>` to filter- `:cd <path>` - Change directory (alias for :project)openaicli tts "Story time" --voice nova --output story.mp3openaicli image "sunset over mountains"# Images

- **Project Switching**: Jump between codebases with `:cd` or `:project`

- **Context Injection**: AI gets full code context for detailed analysis

- **Smart Filtering**: Ignores node_modules, .git, venv, etc.

- **20+ Languages**: Python, JS/TS, Java, C++, Go, Rust, PHP, and more**Example YOLO workflow:**



### Qwen-Inspired Features```

- **Colon Commands**: Use `:help` instead of `/help`

- **Screen Clearing**: `:clear` clears screen and shows welcome:yolo                           # Enable YOLO mode# List modelsopenaicli image "logo design" --model dall-e-3 --size 1792x1024openaicli image "sunset over mountains"

- **Random Seed**: `:seed` for reproducible outputs

- **Runtime Config**: `:conf key=value` without restarting:project ~/myapp                # Switch to your project

- **Organized Output**: Generated files in `generated/` folders

:analyze                        # Scan the codebaseopenaicli models

## Input Tips

# Now ask questions about your code!

- Type normally for single-line input

- End line with `\` for multiline input"What does the authentication system do?"openaicli models --type chatopenaicli image "logo design" --model dall-e-3 --size 1792x1024

- Press `Ctrl+D` to enter multiline mode

- Press `Ctrl+C` to interrupt (won't exit)"Find all API endpoints in this project"



## Examples"Explain the database schema"openaicli models --type image



```bash```

# Start chat with system prompt

openaicli chat --system "You are a helpful coding assistant"```# Video



# Generate image with specific settings### Advanced

openaicli image "cyberpunk city" --model dall-e-3 --quality hd --size 1792x1024

- `:seed` - Show current random seed

# Generate video

openaicli video "time-lapse of sunset over ocean"- `:seed 42` - Set random seed for reproducibility



# Text-to-speech with custom voice- `:model` - Show current model## Interactive Chat Commandsopenaicli video "ocean waves"# Video

openaicli tts "Welcome to MapleCLI" --voice nova --speed 1.2



# YOLO mode for code review

openaicli chat## Configuration

:yolo

:cd ~/my-project

:analyze

# Ask: "Review this codebase for security issues"First run asks for API base URL and key. Saved to `~/.config/openaicli/config.json`When in chat mode, use these commands (with `:` or `/` prefix):openaicli video "fireworks display" --model sora-2openaicli video "ocean waves"

# Ask: "What frameworks and libraries are used?"

# Ask: "Explain how authentication works"

```

Or set environment variables:

## Use Cases for YOLO Mode

```bash

1. **Code Review**: "Review this codebase for security vulnerabilities"

2. **Documentation**: "Generate API documentation from the code"export OPENAI_API_BASE="https://api.openai.com/v1"### Essential Commandsopenaicli video "fireworks display" --model sora-2

3. **Debugging**: "Find potential bugs in the authentication system"

4. **Learning**: "Explain how this React app handles state management"export OPENAI_API_KEY="sk-your-key"

5. **Refactoring**: "Suggest improvements for this component"

6. **Architecture**: "Diagram the overall architecture of this project"```- `:help` or `:h` - Show help message



## Generated Files



All generated content is organized:## Shortcuts- `:exit` or `:q` - Exit chat session# Text-to-Speech

- Images → `generated/images/`

- Videos → `generated/videos/`

- Audio → `generated/audio/`

Add to PowerShell `$PROFILE`:- `:clear` or `:cl` - Clear screen

## Language Preference



By default, responses are in English. To change:

```bash```powershell- `:clear-history` or `:clh` - Clear conversation historyopenaicli tts "Hello world"# Text-to-Speech

openaicli chat --lang zh  # Chinese

openaicli chat --lang ja  # Japanesefunction maple {

openaicli chat --lang es  # Spanish

```    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"


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
