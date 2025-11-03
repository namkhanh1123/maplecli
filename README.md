# MapleCLI

A secure, feature-rich command-line interface for OpenAI-compatible APIs with advanced code analysis capabilities. 🚀

## Features

✨ **Multi-modal Support**: Chat, image generation, video generation, and text-to-speech  
🎨 **Rich Terminal UI**: Beautiful console output with Rich library  
⚙️ **Interactive Commands**: Qwen-style commands with `:` or `/` prefix  
🔧 **Runtime Configuration**: Change temperature, tokens, seed on-the-fly  
💾 **Session Management**: Save/load conversations  
🎲 **Random Seed Control**: Reproducible outputs  
🚀 **YOLO Mode**: Analyze entire codebases and get AI-powered insights!  
📁 **Project Switching**: Navigate between different projects seamlessly  
🔒 **Enhanced Security**: Path traversal protection, secure API key handling  
📊 **Advanced Analysis**: Dependency extraction, complexity metrics, architecture patterns  

## Installation

### Quick Install
```bash
pip install -e .
```

### Development Install
```bash
git clone https://github.com/maplecli/maplecli.git
cd maplecli
pip install -e ".[dev]"
```

## Usage

### Chat
```bash
# Start interactive chat
maplecli chat

# With specific model and settings
maplecli chat --model gpt-4 --temperature 0.9 --system "You are a helpful coding assistant"
```

### Image Generation
```bash
# Generate images
maplecli image "sunset over mountains"
maplecli image "logo design" --model dall-e-3 --size 1792x1024
```

### Video Generation
```bash
# Generate videos
maplecli video "ocean waves"
maplecli video "fireworks display" --model sora-2
```

### Text-to-Speech
```bash
# Convert text to speech
maplecli tts "Hello world"
maplecli tts "Story time" --voice nova --output story.mp3
```

### List Models
```bash
# List available models
maplecli models
maplecli models --type chat
maplecli models --type image
```

## Interactive Chat Commands

When in chat mode, use these commands (with `:` or `/` prefix):

### Essential Commands
- `:help` or `:h` - Show help message
- `:exit` or `:q` - Exit chat session
- `:clear` or `:cl` - Clear screen
- `:clear-history` or `:clh` - Clear conversation history

### Session Management
- `:save <file>` - Save conversation to file
- `:load <file>` - Load conversation from file
- `:history` or `:his` - Show conversation history

### Configuration
- `:conf` - Show current configuration
- `:conf temperature=0.8` - Set temperature
- `:conf max_tokens=1000` - Set max tokens
- `:reset-conf` - Reset to defaults

### YOLO Mode (Code Analysis) 🚀
- `:yolo` - Toggle YOLO mode on/off
- `:analyze` - Scan and analyze current project (auto-reads key files)
- `:read <file>` - Read a specific file and add to context
- `:files` - List all files in current project
- `:files <pattern>` - Filter files by pattern (e.g., `:files tsx`)
- `:project` - Show current project path and recent projects
- `:project <path>` - Switch to a different project
- `:cd <path>` - Change directory (alias for :project)

### Advanced
- `:seed` - Show current random seed
- `:seed 42` - Set random seed for reproducibility
- `:model` - Show current model

## YOLO Mode - Advanced Code Analysis

### Example Workflow
```
:yolo                           # Enable YOLO mode
:cd ~/my-project                # Switch to your project
:analyze                        # Scan the codebase
"What does this codebase do?"   # Ask questions about code
```

### What :analyze Does
- Scans entire project structure
- Counts lines by file type
- Auto-reads up to 10 important files (README, package.json, main files, etc.)
- Injects all code into AI context
- Provides dependency analysis
- Calculates complexity metrics
- Identifies architectural patterns

### Advanced Analysis Features
- **Dependency Extraction**: Automatically identifies imports and dependencies
- **Complexity Metrics**: Cyclomatic and cognitive complexity analysis
- **Architecture Detection**: Identifies MVC, microservices, serverless patterns
- **Security Scanning**: Path traversal protection and secure file handling
- **Performance Monitoring**: Memory usage tracking and file size limits

## Configuration

### First-time Setup
First run asks for API base URL and key. Saved to platform-specific secure location:
- **Windows**: `%APPDATA%\maplecli\config.json`
- **macOS**: `~/Library/Application Support/maplecli/config.json`
- **Linux**: `~/.config/maplecli/config.json`

### Environment Variables
```bash
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-your-key"
```

### Security Features
- **Secure API Key Input**: Hidden input using `getpass`
- **Path Traversal Protection**: Prevents access outside project directory
- **File Size Limits**: 10MB per file, 100MB total project size
- **Atomic Config Updates**: Safe configuration file operations
- **Cross-platform Permissions**: Proper file permissions on all platforms

## Shortcuts

### PowerShell Functions
Add to your PowerShell `$PROFILE`:
```powershell
function maple {
    $env:OPENAI_API_BASE = "https://api.mapleai.de/v1"
    $env:OPENAI_API_KEY = "your-key"
    maplecli @args
}

function gemini {
    $env:OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
    $env:OPENAI_API_KEY = "your-key"
    maplecli @args
}
```

Usage:
```powershell
maple chat --model deepseek-v3.2-exp
gemini chat --model gemini-2.5-pro
```

## Examples

### Basic Chat with System Prompt
```bash
maplecli chat --system "You are a helpful coding assistant"
```

### Image Generation with Custom Settings
```bash
maplecli image "cyberpunk city" --model dall-e-3 --quality hd --size 1792x1024
```

### Video Generation
```bash
maplecli video "time-lapse of sunset over ocean"
```

### Text-to-Speech with Custom Voice
```bash
maplecli tts "Welcome to MapleCLI" --voice nova --speed 1.2
```

### Code Analysis Workflow
```bash
maplecli chat
:yolo
:cd ~/my-project
:analyze
"Review this codebase for security issues"
"Explain the architecture"
"Find all API endpoints"
```

## Security Considerations

- **API Key Protection**: Keys are stored with restricted permissions
- **Input Validation**: All file paths are validated and sanitized
- **Memory Limits**: Prevents resource exhaustion attacks
- **Error Handling**: Comprehensive error logging without exposing sensitive data
- **Audit Logging**: Security events are logged for monitoring

## Development

### Setup Development Environment
```bash
git clone https://github.com/maplecli/maplecli.git
cd maplecli
pip install -e ".[dev,security]"
```

### Run Tests
```bash
pytest
pytest --cov=maplecli
```

### Code Quality
```bash
black main.py
flake8 main.py
mypy main.py
bandit main.py
```

## Architecture

### Core Components
- **ConfigManager**: Secure configuration management
- **ChatClient**: API communication with retry logic
- **CodeAnalyzer**: Advanced code analysis with security
- **CLI**: Command-line interface and user interaction

### Security Features
- **Path Validation**: Prevents directory traversal attacks
- **Size Limits**: Memory and file size protection
- **Error Handling**: Secure error reporting
- **Logging**: Comprehensive audit trails

## Performance

### Optimizations
- **Async Operations**: Non-blocking file operations
- **Memory Management**: Bounded memory usage
- **Progress Tracking**: Real-time progress indicators
- **Caching**: Intelligent file analysis caching

### Limits
- **Max File Size**: 10MB per file
- **Max Project Size**: 100MB total
- **Max Depth**: 10 directory levels
- **Max Files**: 1000 files per analysis

## Troubleshooting

### Common Issues

**Permission Denied**
```bash
# On Unix/Linux/macOS
chmod +x install.sh

# On Windows
# Run PowerShell as Administrator
```

**Module Not Found**
```bash
pip install -e .
# or
pip install rich requests aiofiles
```

**Configuration Issues**
```bash
# Reset configuration
rm -rf ~/.config/maplecli  # Linux
rm -rf ~/Library/Application\ Support/maplecli  # macOS
rm -rf %APPDATA%\maplecli  # Windows
```

### Debug Mode
```bash
# Enable debug logging
export MAPLECLI_DEBUG=1
maplecli chat
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include comprehensive error handling
- Add security considerations for new features
- Update documentation for changes

## License

MIT License - see LICENSE file for details.

## Changelog

### v2.0.0
- 🔒 Enhanced security with path traversal protection
- 📊 Advanced code analysis algorithms
- 🚀 Async operations for better performance
- 📈 Memory management and size limits
- 🛠️ Cross-platform compatibility improvements
- 📝 Comprehensive error handling and logging

### v1.0.0
- 🎉 Initial release
- 💬 Basic chat functionality
- 🖼️ Image generation
- 🎥 Video generation
- 🗣️ Text-to-speech
- 📁 YOLO mode code analysis

## Support

- **Documentation**: https://maplecli.readthedocs.io/
- **Issues**: https://github.com/maplecli/maplecli/issues
- **Discussions**: https://github.com/maplecli/maplecli/discussions
- **Email**: team@maplecli.dev

---

**MapleCLI** - Secure, powerful, and intelligent AI interaction for developers. 🚀
