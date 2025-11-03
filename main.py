import argparse
import os
import requests
import sys
import json
import stat
import platform
import random
import pathlib
import logging
import getpass
import hashlib
import asyncio
import aiofiles
from typing import List, Dict, Optional, Tuple, Set, Union, AsyncIterator
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Security and logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maplecli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("maplecli")

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

class MapleLogger:
    """Enhanced logging with structured error context"""
    def __init__(self, name: str = "maplecli"):
        self.logger = logging.getLogger(name)
        
    def log_error(self, error: Exception, operation: str, filepath: Optional[str] = None, severity: str = "medium"):
        """Structured error logging"""
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            extra={
                'severity': severity,
                'filepath': filepath,
                'error_type': type(error).__name__
            }
        )
        
    def log_security_event(self, event: str, details: str):
        """Log security-related events"""
        self.logger.warning(f"SECURITY: {event} - {details}")

maple_logger = MapleLogger()

WELCOME_MSG = """\
[bold cyan]╔════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]  [bold white]MapleCLI[/bold white] - OpenAI Compatible Chat Interface      [bold cyan]║[/bold cyan]
[bold cyan]╚════════════════════════════════════════════════════════════╝[/bold cyan]

[dim]→[/dim] Type your message to chat with the AI
[dim]→[/dim] Type [yellow]:help[/yellow] to see all commands
[dim]→[/dim] Type [yellow]:yolo[/yellow] to enable code analysis mode

[dim]Press Ctrl+C to interrupt • Ctrl+D for multiline input[/dim]
"""

HELP_MSG = """\
[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]
[bold white]                      Available Commands                       [/bold white]
[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]

[bold yellow]Chat Commands:[/bold yellow]
  [cyan]:help, :h[/cyan]                Show this help
  [cyan]:exit, :quit, :q[/cyan]         Exit chat
  [cyan]:clear, :cl[/cyan]              Clear screen
  [cyan]:history, :his[/cyan]           Show chat history
  [cyan]:clear-history, :clh[/cyan]     Clear history

[bold yellow]Configuration:[/bold yellow]
  [cyan]:conf[/cyan]                    Show current settings
  [cyan]:conf <key>=<value>[/cyan]      Change settings
  [cyan]:model[/cyan]                   Show current model
  
  [dim]Examples: :conf temperature=0.8, :conf max_tokens=2000[/dim]

[bold yellow]YOLO Mode - Code Analysis:[/bold yellow]
  [cyan]:yolo[/cyan]                    Toggle code analysis
  [cyan]:analyze[/cyan]                 Scan project structure
  [cyan]:read <file>[/cyan]             Read specific file
  [cyan]:files [pattern][/cyan]         List/filter project files
  [cyan]:cd <path>[/cyan]               Switch project directory

[bold yellow]Save & Load:[/bold yellow]
  [cyan]:save <file>[/cyan]             Save conversation
  [cyan]:load <file>[/cyan]             Load conversation

[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]
[dim]Tip: End line with \\ for multiline or press Ctrl+D[/dim]
"""

class ConfigManager:
    """Manages the configuration for the CLI."""
    def __init__(self, console: Console):
        self.console = console
        self.config_dir = os.path.expanduser("~/.config/openaicli")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.api_base: Optional[str] = None
        self.api_key: Optional[str] = None
        self.recent_projects: List[str] = []

    def load_config(self) -> None:
        """Loads configuration from file or environment variables."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                try:
                    config = json.load(f)
                    self.api_base = config.get("OPENAI_API_BASE")
                    self.api_key = config.get("OPENAI_API_KEY")
                    self.recent_projects = config.get("recent_projects", [])
                except json.JSONDecodeError as e:
                    self.console.print(f"[bold yellow]Warning: Config file corrupted ({e}). Using environment variables or prompting.[/bold yellow]")

        if not self.api_base:
            self.api_base = os.environ.get("OPENAI_API_BASE")
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_base:
            self.console.print("[bold red]OpenAI API base URL not found.[/bold red]")
            self.api_base = input("Please enter your OpenAI API base URL: ")
            # Use secure input for API key
            try:
                self.api_key = getpass.getpass("Please enter your OpenAI API key (input hidden, optional): ")
            except Exception as e:
                maple_logger.log_error(e, "API key input", severity="high")
                self.console.print("[yellow]Warning: Could not hide input. API key will be visible.[/yellow]")
                self.api_key = input("Please enter your OpenAI API key (optional): ")
            self.save_config()

    def save_config(self) -> None:
        """Saves API configuration to the config file with enhanced security."""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Create config data
            config_data = {
                "OPENAI_API_BASE": self.api_base,
                "OPENAI_API_KEY": self.api_key,
                "recent_projects": self.recent_projects[-10:]  # Keep last 10 projects
            }
            
            # Write to temporary file first (atomic operation)
            temp_file = self.config_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Atomic rename
            if os.path.exists(self.config_file):
                os.replace(temp_file, self.config_file)
            else:
                os.rename(temp_file, self.config_file)
            
            # Set restrictive permissions on config file
            self._secure_config_file(self.config_file)
            
            self.console.print(f"Configuration saved to [green]{self.config_file}[/green]")
            maple_logger.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            maple_logger.log_error(e, "save_config", self.config_file, severity="high")
            self.console.print(f"[bold red]Error saving configuration: {e}[/bold red]")
    
    def _secure_config_file(self, filepath: str) -> None:
        """Set secure permissions across platforms"""
        try:
            if platform.system() != "Windows":
                os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)
            else:
                # On Windows, we can try to set file attributes
                import ctypes
                try:
                    # Hide the file on Windows
                    ctypes.windll.kernel32.SetFileAttributesW(filepath, 0x2)  # FILE_ATTRIBUTE_HIDDEN
                except:
                    pass
        except Exception as e:
            maple_logger.log_error(e, "secure_config_file", filepath, severity="medium")
    
    def add_recent_project(self, project_path: str) -> None:
        """Adds a project to the recent projects list."""
        abs_path = os.path.abspath(project_path)
        if abs_path in self.recent_projects:
            self.recent_projects.remove(abs_path)
        self.recent_projects.append(abs_path)
        self.save_config()

class ChatClient:
    """Handles all communication with the OpenAI-compatible API."""
    def __init__(self, api_base: str, api_key: Optional[str], console: Console):
        self.api_base = api_base
        self.api_key = api_key
        self.console = console
        self.timeout = 30  # Default timeout in seconds
        self.max_retries = 3

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, any]]:
        """Fetches the list of available models from the API.
        
        Args:
            model_type: Filter models by type (e.g., '/v1/chat/completions', '/v1/images/generations')
        
        Returns:
            List of model dictionaries with id, type, and owned_by fields
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with self.console.status("[bold green]Fetching models...[/bold green]"):
                response = requests.get(f"{self.api_base}/models", headers=headers, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    models = []
                    for model in data["data"]:
                        if "id" in model:
                            # Filter by type if specified
                            if model_type:
                                model_types = model.get("type", [])
                                if isinstance(model_types, list) and model_type in model_types:
                                    models.append(model)
                            else:
                                models.append(model)
                    return models
                else:
                    self.console.print("[bold red]Error: Unexpected API response format[/bold red]")
                    return []
            else:
                self.console.print(f"[bold red]Error: Failed to fetch models with status code {response.status_code}[/bold red]")
                return []
        except requests.exceptions.Timeout:
            self.console.print(f"[bold red]Error: Request timed out after {self.timeout} seconds[/bold red]")
            return []
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return []

    def stream_chat(self, history: List[Dict[str, str]], model: str, 
                    temperature: float = 0.7, max_tokens: Optional[int] = None) -> Tuple[List[Dict[str, str]], str]:
        """Sends a prompt to the chat API and streams the response."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": model,
            "messages": history,
            "stream": True,
            "temperature": temperature
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        assistant_response = ""
        
        # Show a clean thinking indicator
        print()
        self.console.print("[bold white]┌─[/bold white][bold cyan]Assistant[/bold cyan][bold white]:[/bold white]")
        self.console.print("[bold white]│[/bold white] [dim]Thinking...[/dim]", end="")
        
        for attempt in range(self.max_retries):
            try:
                with requests.post(f"{self.api_base}/chat/completions", 
                                 headers=headers, json=data, stream=True, 
                                 timeout=self.timeout) as response:
                    if response.status_code == 200:
                        # Clear the thinking line
                        print("\r" + " " * 80 + "\r", end="", flush=True)
                        self.console.print("[bold white]└─>[/bold white] ", end="")
                        
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith("data: "):
                                    chunk_str = chunk_str[6:]
                                if chunk_str == "[DONE]":
                                    break
                                try:
                                    json_chunk = json.loads(chunk_str)
                                    if "choices" in json_chunk and len(json_chunk["choices"]) > 0:
                                        choice = json_chunk["choices"][0]
                                        if "delta" in choice and "content" in choice["delta"]:
                                            content = choice["delta"]["content"]
                                            assistant_response += content
                                            # Print content as it arrives
                                            print(content, end="", flush=True)
                                except json.JSONDecodeError as e:
                                    # Log parsing errors but continue streaming
                                    if chunk_str and chunk_str != "[DONE]":
                                        self.console.print(f"\n[dim yellow]Warning: Failed to parse chunk: {e}[/dim yellow]")
                        
                        print()  # New line after streaming
                        
                        if not assistant_response:
                            self.console.print("[bold yellow]Warning: Received empty response from API[/bold yellow]")
                        
                        break  # Success, exit retry loop
                    else:
                        error_message = f"API request failed with status code {response.status_code}"
                        try:
                            error_detail = response.json()
                            if "error" in error_detail:
                                error_message += f": {error_detail['error']}"
                        except:
                            error_message += f": {response.text[:200]}"
                        
                        self.console.print(f"\n[bold red]Error: {error_message}[/bold red]")
                        
                        if attempt < self.max_retries - 1:
                            self.console.print(f"[yellow]Retrying ({attempt + 1}/{self.max_retries})...[/yellow]")
                        else:
                            raise Exception(error_message)

            except requests.exceptions.Timeout:
                self.console.print(f"\n[bold red]Error: Request timed out after {self.timeout} seconds[/bold red]")
                if attempt < self.max_retries - 1:
                    self.console.print(f"[yellow]Retrying ({attempt + 1}/{self.max_retries})...[/yellow]")
                else:
                    raise
            except requests.exceptions.RequestException as e:
                self.console.print(f"\n[bold red]Error: {e}[/bold red]")
                if attempt < self.max_retries - 1:
                    self.console.print(f"[yellow]Retrying ({attempt + 1}/{self.max_retries})...[/yellow]")
                else:
                    raise
        
        history.append({"role": "assistant", "content": assistant_response})
        return history, assistant_response

    def generate_image(self, prompt: str, model: str = "dall-e-3", 
                      size: str = "1024x1024", quality: str = "standard",
                      n: int = 1) -> List[str]:
        """Generates images from text prompt.
        
        Args:
            prompt: Text description of the image to generate
            model: Model to use for generation
            size: Image size (e.g., "1024x1024", "1792x1024", "1024x1792")
            quality: Image quality ("standard" or "hd")
            n: Number of images to generate
            
        Returns:
            List of image URLs or base64 data
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality
        }

        try:
            self.console.print("[bold cyan]Generating image...[/bold cyan]")
            response = requests.post(
                f"{self.api_base}/images/generations",
                headers=headers,
                json=data,
                timeout=60  # Longer timeout for image generation
            )

            if response.status_code == 200:
                result = response.json()
                if "data" in result:
                    urls = []
                    for img in result["data"]:
                        if "url" in img:
                            urls.append(img["url"])
                        elif "b64_json" in img:
                            urls.append(img["b64_json"])
                    return urls
                else:
                    self.console.print("[bold red]Error: No image data in response[/bold red]")
                    return []
            else:
                error_msg = f"Image generation failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                self.console.print(f"[bold red]{error_msg}[/bold red]")
                return []

        except requests.exceptions.Timeout:
            self.console.print("[bold red]Error: Image generation timed out[/bold red]")
            return []
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return []

    def generate_video(self, prompt: str, model: str = "sora-2") -> Optional[str]:
        """Generates video from text prompt.
        
        Args:
            prompt: Text description of the video to generate
            model: Model to use for generation
            
        Returns:
            Video URL or base64 data
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": model,
            "prompt": prompt
        }

        try:
            self.console.print("[bold cyan]Generating video (this may take a while)...[/bold cyan]")
            response = requests.post(
                f"{self.api_base}/videos",
                headers=headers,
                json=data,
                timeout=300  # 5 minute timeout for video
            )

            if response.status_code == 200:
                result = response.json()
                if "url" in result:
                    return result["url"]
                elif "data" in result and len(result["data"]) > 0:
                    if "url" in result["data"][0]:
                        return result["data"][0]["url"]
                    elif "b64" in result["data"][0]:
                        return result["data"][0]["b64"]
                self.console.print("[bold red]Error: No video data in response[/bold red]")
                return None
            else:
                error_msg = f"Video generation failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                self.console.print(f"[bold red]{error_msg}[/bold red]")
                return None

        except requests.exceptions.Timeout:
            self.console.print("[bold red]Error: Video generation timed out[/bold red]")
            return None
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return None

    def text_to_speech(self, text: str, model: str = "tts-1", 
                       voice: str = "alloy", speed: float = 1.0) -> Optional[bytes]:
        """Converts text to speech.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speech speed (0.25 to 4.0)
            
        Returns:
            Audio data as bytes
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed
        }

        try:
            self.console.print("[bold cyan]Generating speech...[/bold cyan]")
            response = requests.post(
                f"{self.api_base}/audio/speech",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.content
            else:
                error_msg = f"TTS failed with status {response.status_code}"
                self.console.print(f"[bold red]{error_msg}[/bold red]")
                return None

        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return None

class CodeAnalyzer:
    """
    Analyzes source code in a project directory.
    
    Inspired by Qwen Code and Gemini CLI's workspace context management:
    - Respects .gitignore patterns
    - Provides structured folder hierarchy
    - Intelligent file discovery and filtering
    - Context-aware code reading
    """
    
    # Common source code extensions
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        '.m', '.mm', '.sh', '.bash', '.zsh', '.bat', '.cmd', '.ps1', '.sql', '.html', '.css', '.scss',
        '.sass', '.vue', '.svelte', '.json', '.yaml', '.yml', '.xml', '.md',
        '.toml', '.ini', '.cfg', '.conf', '.dockerfile', '.lock', '.txt'
    }
    
    # Directories to ignore (following Qwen/Gemini patterns)
    IGNORE_DIRS = {
        'node_modules', '.git', '.venv', 'venv', 'env', '__pycache__',
        '.pytest_cache', 'dist', 'build', 'target', '.idea', '.vscode',
        'vendor', 'tmp', 'temp', '.cache', 'coverage', '.next', 'out',
        '.angular', '.svelte-kit', '.nuxt', '.output', 'bower_components',
        '.dart_tool', 'buck-out', '.gradle', '.mvn'
    }
    
    # File patterns that indicate project type
    PROJECT_MARKERS = {
        'python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
        'node': ['package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml'],
        'java': ['pom.xml', 'build.gradle', 'build.gradle.kts'],
        'rust': ['Cargo.toml', 'Cargo.lock'],
        'go': ['go.mod', 'go.sum'],
        'php': ['composer.json', 'composer.lock'],
        'ruby': ['Gemfile', 'Gemfile.lock'],
        '.net': ['*.csproj', '*.sln', 'packages.config']
    }
    
    def __init__(self, project_path: str, console: Console):
        self.project_path = os.path.abspath(project_path)
        self.console = console
        self.files: List[str] = []
        self.file_stats: Dict[str, int] = {}
        self.total_lines = 0
        self.project_type: Optional[str] = None
        self.key_files: List[str] = []  # Important files like README, config files
        
        # Performance and security limits
        self.max_file_size = 10 * 1024 * 1024  # 10MB per file
        self.max_total_size = 100 * 1024 * 1024  # 100MB total
        self.max_depth = 10  # Maximum directory depth
        self.processed_size = 0  # Track total processed size
        
    def _should_ignore_file(self, filepath: str) -> bool:
        """
        Check if file should be ignored (similar to Gemini/Qwen's gitignore filtering).
        
        In production, this would use proper gitignore parsing library.
        For now, basic pattern matching.
        """
        basename = os.path.basename(filepath)
        
        # Ignore hidden files except specific ones
        if basename.startswith('.') and basename not in {'.env.example', '.gitignore', '.dockerignore'}:
            return True
        
        # Ignore common generated/lock files
        ignore_patterns = ['*.lock', '*.log', '*.tmp', '*.swp', '*.bak', '*.pyc', '*.class']
        for pattern in ignore_patterns:
            if pattern.replace('*', '') in basename:
                return True
        
        return False
    
    def _detect_project_type(self):
        """Detect project type based on marker files (like Qwen's project detection)."""
        for project_type, markers in self.PROJECT_MARKERS.items():
            for marker in markers:
                marker_path = os.path.join(self.project_path, marker)
                if os.path.exists(marker_path):
                    self.project_type = project_type
                    return
        self.project_type = 'unknown'
    
    async def scan_project_async(self) -> Dict[str, any]:
        """
        Asynchronously scans the project directory with progress tracking and memory management.
        
        Enhanced version with security, performance, and error handling improvements.
        """
        if not os.path.exists(self.project_path):
            self.console.print(f"[bold red]Error: Path does not exist: {self.project_path}[/bold red]")
            return {}
        
        if not os.path.isdir(self.project_path):
            self.console.print(f"[bold red]Error: Path is not a directory: {self.project_path}[/bold red]")
            return {}
        
        # Reset state
        self.files = []
        self.file_stats = {}
        self.total_lines = 0
        self.key_files = []
        self.processed_size = 0
        skipped_files = []
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Scanning project...", total=None)
            
            try:
                # Detect project type first
                self._detect_project_type()
                progress.update(task, description="[cyan]Detecting project type...")
                
                # Scan for files with progress tracking
                file_count = 0
                for root, dirs, files in os.walk(self.project_path):
                    # Filter out ignored directories (in-place modification)
                    dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                    
                    # Calculate relative path for depth limiting
                    rel_root = os.path.relpath(root, self.project_path)
                    depth = len(rel_root.split(os.sep)) if rel_root != '.' else 0
                    
                    # Limit depth to avoid overwhelming scans
                    if depth > self.max_depth:
                        dirs.clear()
                        continue
                    
                    for file in files:
                        file_count += 1
                        progress.update(task, description=f"[cyan]Scanning... {file_count} files")
                        
                        # Check if should ignore
                        if self._should_ignore_file(file):
                            skipped_files.append(file)
                            continue
                        
                        ext = os.path.splitext(file)[1].lower()
                        filepath = os.path.join(root, file)
                        rel_path = os.path.relpath(filepath, self.project_path)
                        
                        # Security: Check file size
                        try:
                            file_size = os.path.getsize(filepath)
                            if file_size > self.max_file_size:
                                maple_logger.log_security_event(
                                    "Large file skipped",
                                    f"File: {rel_path}, Size: {file_size} bytes"
                                )
                                skipped_files.append(f"{rel_path} (too large)")
                                continue
                                
                            # Check total size limit
                            if self.processed_size + file_size > self.max_total_size:
                                maple_logger.log_security_event(
                                    "Size limit reached",
                                    f"Current: {self.processed_size}, File: {file_size}"
                                )
                                skipped_files.append(f"{rel_path} (size limit)")
                                continue
                                
                            self.processed_size += file_size
                            
                        except Exception as e:
                            maple_logger.log_error(e, "file_size_check", rel_path)
                            skipped_files.append(f"{rel_path} (size check failed)")
                            continue
                        
                        # Identify key files (README, configs, etc.)
                        if any(keyword in file.upper() for keyword in ['README', 'LICENSE', 'CONTRIBUTING']):
                            self.key_files.append(rel_path)
                        elif file in ['package.json', 'tsconfig.json', 'vite.config.ts', 'vite.config.js',
                                      'setup.py', 'pyproject.toml', 'Cargo.toml', 'go.mod', 'main.py',
                                      'main.ts', 'index.ts', 'index.js', 'App.tsx', 'App.vue']:
                            self.key_files.append(rel_path)
                        
                        # Track code files
                        if ext in self.CODE_EXTENSIONS:
                            self.files.append(rel_path)
                            
                            # Count lines with error handling
                            try:
                                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = sum(1 for _ in f)
                                    self.total_lines += lines
                                    self.file_stats[ext] = self.file_stats.get(ext, 0) + lines
                            except Exception as e:
                                maple_logger.log_error(e, "line_count", rel_path)
                                skipped_files.append(f"{rel_path} (read error)")
                                continue
                
                progress.update(task, description="[green]Scan complete!", completed=True)
                
            except Exception as e:
                maple_logger.log_error(e, "scan_project", self.project_path, severity="high")
                self.console.print(f"[bold red]Error during scan: {e}[/bold red]")
        
        return {
            'project_path': self.project_path,
            'project_type': self.project_type,
            'total_files': len(self.files),
            'total_lines': self.total_lines,
            'file_stats': self.file_stats,
            'key_files': self.key_files,
            'files': self.files[:100],  # Limit to first 100 for display
            'skipped_files': skipped_files,
            'processed_size_mb': round(self.processed_size / (1024 * 1024), 2)
        }
    
    def scan_project(self) -> Dict[str, any]:
        """Synchronous wrapper for scan_project_async"""
        try:
            return asyncio.run(self.scan_project_async())
        except Exception as e:
            maple_logger.log_error(e, "scan_project_sync", self.project_path)
            return {}
    
    def get_project_tree(self, max_depth: int = 3) -> Tree:
        """Creates a visual tree representation of the project structure."""
        tree = Tree(f"[bold cyan]{os.path.basename(self.project_path)}[/bold cyan]")
        
        def add_to_tree(parent_tree: Tree, path: str, current_depth: int):
            if current_depth >= max_depth:
                return
            
            try:
                items = sorted(os.listdir(path))
            except PermissionError:
                return
            
            dirs = [item for item in items if os.path.isdir(os.path.join(path, item)) and item not in self.IGNORE_DIRS]
            files = [item for item in items if os.path.isfile(os.path.join(path, item))]
            
            # Limit display
            if len(dirs) + len(files) > 20:
                dirs = dirs[:10]
                files = files[:10]
            
            for d in dirs:
                dir_path = os.path.join(path, d)
                branch = parent_tree.add(f"[blue]{d}/[/blue]")
                add_to_tree(branch, dir_path, current_depth + 1)
            
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in self.CODE_EXTENSIONS:
                    parent_tree.add(f"[green]{f}[/green]")
                else:
                    parent_tree.add(f"[dim]{f}[/dim]")
        
        add_to_tree(tree, self.project_path, 0)
        return tree
    
    def analyze_dependencies(self, filepath: str) -> Dict[str, List[str]]:
        """Extract import/require dependencies from source files."""
        dependencies = {'imports': [], 'requires': []}
        
        try:
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.py':
                dependencies['imports'] = self._analyze_python_dependencies(filepath)
            elif ext in ['.js', '.ts', '.jsx', '.tsx']:
                dependencies['imports'] = self._analyze_javascript_dependencies(filepath)
            elif ext == '.java':
                dependencies['imports'] = self._analyze_java_dependencies(filepath)
            elif ext in ['.go']:
                dependencies['imports'] = self._analyze_go_dependencies(filepath)
            elif ext in ['.rs']:
                dependencies['imports'] = self._analyze_rust_dependencies(filepath)
                
        except Exception as e:
            maple_logger.log_error(e, "analyze_dependencies", filepath)
            
        return dependencies
    
    def _analyze_python_dependencies(self, filepath: str) -> List[str]:
        """Extract Python imports."""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Simple regex-based import extraction
                import re
                patterns = [
                    r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
                    r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                    r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_.]*)\s+import'
                ]
                
                for line in content.split('\n'):
                    for pattern in patterns:
                        matches = re.findall(pattern, line.strip())
                        for match in matches:
                            # Clean up the import name
                            import_name = match.split('.')[0].split(' as ')[0]
                            if import_name and import_name not in imports:
                                imports.append(import_name)
                                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_python_dependencies", filepath)
            
        return imports
    
    def _analyze_javascript_dependencies(self, filepath: str) -> List[str]:
        """Extract JavaScript/TypeScript imports."""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                import re
                patterns = [
                    r'^import\s+.*\s+from\s+["\']([^"\']+)["\']',
                    r'^import\s+["\']([^"\']+)["\']',
                    r'^const\s+.*=\s+require\(["\']([^"\']+)["\']',
                    r'^import\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']'
                ]
                
                for line in content.split('\n'):
                    for pattern in patterns:
                        matches = re.findall(pattern, line.strip())
                        for match in matches:
                            if isinstance(match, tuple):
                                # For patterns with groups
                                import_name = match[1] if len(match) > 1 else match[0]
                            else:
                                import_name = match
                            
                            if import_name and import_name not in imports:
                                imports.append(import_name)
                                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_javascript_dependencies", filepath)
            
        return imports
    
    def _analyze_java_dependencies(self, filepath: str) -> List[str]:
        """Extract Java imports."""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                import re
                pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_.]*)'
                
                for line in content.split('\n'):
                    matches = re.findall(pattern, line.strip())
                    for match in matches:
                        import_name = match.split('.')[0]
                        if import_name and import_name not in imports:
                            imports.append(import_name)
                            
        except Exception as e:
            maple_logger.log_error(e, "_analyze_java_dependencies", filepath)
            
        return imports
    
    def _analyze_go_dependencies(self, filepath: str) -> List[str]:
        """Extract Go imports."""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                import re
                pattern = r'^"([^"]+)"\s*$'
                
                for line in content.split('\n'):
                    if line.strip().startswith('import '):
                        match = re.search(pattern, line.strip())
                        if match:
                            import_name = match.group(1)
                            if import_name and import_name not in imports:
                                imports.append(import_name)
                                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_go_dependencies", filepath)
            
        return imports
    
    def _analyze_rust_dependencies(self, filepath: str) -> List[str]:
        """Extract Rust dependencies."""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                import re
                patterns = [
                    r'^use\s+([a-zA-Z_][a-zA-Z0-9_::]*)',
                    r'^extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                ]
                
                for line in content.split('\n'):
                    for pattern in patterns:
                        matches = re.findall(pattern, line.strip())
                        for match in matches:
                            import_name = match.split('::')[0]
                            if import_name and import_name not in imports:
                                imports.append(import_name)
                                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_rust_dependencies", filepath)
            
        return imports
    
    def detect_architecture_patterns(self, files: List[str]) -> Dict[str, int]:
        """Identify architectural patterns (MVC, microservices, etc.)."""
        patterns = {
            'mvc': 0,
            'microservices': 0,
            'serverless': 0,
            'monolith': 0,
            'modular': 0,
            'event_driven': 0,
            'repository': 0
        }
        
        # Check for MVC pattern
        mvc_indicators = ['controller', 'model', 'view', 'models/', 'views/', 'controllers/']
        for indicator in mvc_indicators:
            if any(indicator in f.lower() for f in files):
                patterns['mvc'] += 1
        
        # Check for microservices
        microservice_indicators = ['service-', '-service/', 'services/', 'api/', 'gateway/']
        for indicator in microservice_indicators:
            if any(indicator in f.lower() for f in files):
                patterns['microservices'] += 1
        
        # Check for serverless
        serverless_indicators = ['functions/', 'lambda/', 'api/', 'netlify/', 'vercel/']
        for indicator in serverless_indicators:
            if any(indicator in f.lower() for f in files):
                patterns['serverless'] += 1
        
        # Check for modular architecture
        modular_indicators = ['modules/', 'components/', 'lib/', 'utils/', 'common/']
        for indicator in modular_indicators:
            if any(indicator in f.lower() for f in files):
                patterns['modular'] += 1
        
        # Check for repository pattern
        repo_indicators = ['repository', 'repositories/', 'dao/', 'persistence/']
        for indicator in repo_indicators:
            if any(indicator in f.lower() for f in files):
                patterns['repository'] += 1
        
        return patterns
    
    def analyze_complexity(self, filepath: str) -> Dict[str, int]:
        """Calculate cyclomatic complexity metrics."""
        complexity = {
            'cyclomatic': 0,
            'cognitive': 0,
            'lines_of_code': 0,
            'functions': 0
        }
        
        try:
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.py':
                complexity = self._analyze_python_complexity(filepath)
            elif ext in ['.js', '.ts']:
                complexity = self._analyze_javascript_complexity(filepath)
            elif ext == '.java':
                complexity = self._analyze_java_complexity(filepath)
                
        except Exception as e:
            maple_logger.log_error(e, "analyze_complexity", filepath)
            
        return complexity
    
    def _analyze_python_complexity(self, filepath: str) -> Dict[str, int]:
        """Calculate Python cyclomatic complexity."""
        complexity = {'cyclomatic': 0, 'cognitive': 0, 'lines_of_code': 0, 'functions': 0}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Count lines of code (excluding comments and empty lines)
                loc = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        loc += 1
                
                complexity['lines_of_code'] = loc
                
                # Count functions
                import re
                function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                functions = re.findall(function_pattern, content)
                complexity['functions'] = len(functions)
                
                # Simple cyclomatic complexity (count decision points)
                decision_points = 0
                decision_keywords = ['if', 'elif', 'for', 'while', 'except', 'and', 'or', 'case']
                for line in lines:
                    stripped = line.strip()
                    for keyword in decision_keywords:
                        if re.search(r'\b' + keyword + r'\b', stripped):
                            decision_points += 1
                
                complexity['cyclomatic'] = decision_points + 1  # Base complexity
                
                # Simple cognitive complexity (nested decision points)
                max_nesting = 0
                current_nesting = 0
                for line in lines:
                    stripped = line.strip()
                    if any(keyword in stripped for keyword in ['if', 'for', 'while', 'try', 'with']):
                        current_nesting += 1
                        max_nesting = max(max_nesting, current_nesting)
                    elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                        continue
                    elif stripped and not stripped.startswith('#'):
                        current_nesting = max(0, current_nesting - 1)
                
                complexity['cognitive'] = max_nesting
                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_python_complexity", filepath)
            
        return complexity
    
    def _analyze_javascript_complexity(self, filepath: str) -> Dict[str, int]:
        """Calculate JavaScript/TypeScript cyclomatic complexity."""
        complexity = {'cyclomatic': 0, 'cognitive': 0, 'lines_of_code': 0, 'functions': 0}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Count lines of code
                loc = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                        loc += 1
                
                complexity['lines_of_code'] = loc
                
                # Count functions
                import re
                function_patterns = [
                    r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                    r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',
                    r'let\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',
                    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\(',
                    r'async\s+function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                ]
                
                functions = []
                for pattern in function_patterns:
                    functions.extend(re.findall(pattern, content))
                
                complexity['functions'] = len(functions)
                
                # Simple cyclomatic complexity
                decision_points = 0
                decision_keywords = ['if', 'else if', 'for', 'while', 'catch', '&&', '||', 'case']
                for line in lines:
                    stripped = line.strip()
                    for keyword in decision_keywords:
                        if keyword in stripped:
                            decision_points += 1
                
                complexity['cyclomatic'] = decision_points + 1
                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_javascript_complexity", filepath)
            
        return complexity
    
    def _analyze_java_complexity(self, filepath: str) -> Dict[str, int]:
        """Calculate Java cyclomatic complexity."""
        complexity = {'cyclomatic': 0, 'cognitive': 0, 'lines_of_code': 0, 'functions': 0}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Count lines of code
                loc = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                        loc += 1
                
                complexity['lines_of_code'] = loc
                
                # Count methods
                import re
                method_pattern = r'(public|private|protected)?\s*(static\s+)?[a-zA-Z_][a-zA-Z0-9_<>\[\]]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                methods = re.findall(method_pattern, content)
                complexity['functions'] = len(methods)
                
                # Simple cyclomatic complexity
                decision_points = 0
                decision_keywords = ['if', 'else if', 'for', 'while', 'catch', '&&', '||', 'case']
                for line in lines:
                    stripped = line.strip()
                    for keyword in decision_keywords:
                        if keyword in stripped:
                            decision_points += 1
                
                complexity['cyclomatic'] = decision_points + 1
                
        except Exception as e:
            maple_logger.log_error(e, "_analyze_java_complexity", filepath)
            
        return complexity
    
    def read_file_content(self, relative_path: str, max_lines: int = 100) -> Optional[str]:
        """Reads the content of a specific file with security validation."""
        try:
            # Security: Validate and sanitize the path
            filepath = self._safe_join_path(self.project_path, relative_path)
            if not filepath or not os.path.exists(filepath):
                return None
            
            # Security: Check file size to prevent memory exhaustion
            file_size = os.path.getsize(filepath)
            max_file_size = 10 * 1024 * 1024  # 10MB limit
            if file_size > max_file_size:
                maple_logger.log_security_event(
                    "Large file access blocked",
                    f"File: {relative_path}, Size: {file_size} bytes"
                )
                return f"Error: File too large ({file_size} bytes > {max_file_size} bytes)"
            
            # Security: Ensure it's a regular file
            if not os.path.isfile(filepath):
                return f"Error: Not a regular file: {relative_path}"
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:max_lines]
                content = ''.join(lines)
                remaining = sum(1 for _ in f)
                if remaining > 0:
                    content += f"\n... ({remaining} more lines)"
                return content
                
        except SecurityError as e:
            maple_logger.log_security_event("Path traversal attempt", f"Path: {relative_path}")
            return f"Security Error: {e}"
        except Exception as e:
            maple_logger.log_error(e, "read_file_content", relative_path)
            return f"Error reading file: {e}"
    
    def _safe_join_path(self, base_path: str, relative_path: str) -> Optional[str]:
        """Prevent path traversal attacks with secure path joining."""
        try:
            # Normalize paths
            base_path = os.path.abspath(base_path)
            relative_path = os.path.normpath(relative_path)
            
            # Check for path traversal attempts
            if '..' in relative_path or relative_path.startswith('/'):
                raise SecurityError("Path traversal detected")
            
            # Join and resolve
            full_path = os.path.abspath(os.path.join(base_path, relative_path))
            
            # Ensure the resolved path is still within base_path
            if not full_path.startswith(base_path):
                raise SecurityError("Path traversal detected")
            
            return full_path
            
        except (ValueError, OSError) as e:
            maple_logger.log_error(e, "safe_join_path", relative_path, severity="high")
            raise SecurityError(f"Invalid path: {e}")
    
    
    def generate_summary(self) -> str:
        """
        Generates a comprehensive text summary for AI context.
        
        Following Qwen Code's getDirectoryContextString pattern:
        provides structured information about the workspace.
        """
        summary_parts = [
            "=== WORKSPACE CONTEXT ===",
            f"\nI'm currently analyzing the directory: {self.project_path}",
            f"Project Name: {os.path.basename(self.project_path)}",
            f"Project Type: {self.project_type or 'unknown'}",
            f"\n=== PROJECT STATISTICS ===",
            f"Total Code Files: {len(self.files)}",
            f"Total Lines of Code: {self.total_lines:,}",
        ]
        
        if self.file_stats:
            summary_parts.append("\n=== LANGUAGES & FILE TYPES ===")
            for ext, lines in sorted(self.file_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (lines / self.total_lines * 100) if self.total_lines > 0 else 0
                summary_parts.append(f"  {ext:10s}: {lines:7,} lines ({percentage:5.1f}%)")
        
        if self.key_files:
            summary_parts.append("\n=== KEY FILES (README, Configs, Entry Points) ===")
            for file in self.key_files[:15]:
                summary_parts.append(f"  - {file}")
        
        summary_parts.append("\n=== PROJECT STRUCTURE (First 40 files) ===")
        summary_parts.append("Here is the folder structure of the workspace:")
        for file in self.files[:40]:
            summary_parts.append(f"  - {file}")
        
        if len(self.files) > 40:
            summary_parts.append(f"  ... and {len(self.files) - 40} more files")
        
        summary_parts.append("\n" + "="*50)
        summary_parts.append("This context helps me understand the codebase structure.")
        summary_parts.append("Use specific file names if you want me to read their contents.")
        summary_parts.append("="*50)
        
        return "\n".join(summary_parts)

class CLI:
    """Handles the command-line interface and chat loop."""
    def __init__(self):
        self.console = Console()
        self.config_manager = ConfigManager(self.console)
        self.chat_client: Optional[ChatClient] = None
        self.max_history_messages = 100  # Limit conversation history
        self.temperature = 0.7
        self.max_tokens: Optional[int] = None
        self.seed: Optional[int] = None  # Random seed for reproducible outputs
        self.yolo_mode = False  # YOLO mode for code analysis
        self.current_project: Optional[str] = None
        self.code_analyzer: Optional[CodeAnalyzer] = None
        
    def clear_screen(self) -> None:
        """Clears the terminal screen."""
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    def run(self) -> None:
        """Runs the main CLI logic."""
        parser = argparse.ArgumentParser(description="A CLI for OpenAI-compatible APIs.")
        subparsers = parser.add_subparsers(dest="command")

        # Chat command
        chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
        chat_parser.add_argument("--model", help="The model to use for the chat.")
        chat_parser.add_argument("--system", help="Set a system prompt for the chat session.")
        chat_parser.add_argument("--lang", "--language", default="en", 
                                help="Preferred response language (en, zh, etc.). Default: en")
        chat_parser.add_argument("--temperature", type=float, default=0.7, 
                                help="Sampling temperature (0.0 to 2.0). Default: 0.7")
        chat_parser.add_argument("--max-tokens", type=int, 
                                help="Maximum tokens in response. Default: unlimited")

        # Image generation command
        image_parser = subparsers.add_parser("image", help="Generate images from text prompts.",
                                            description="Generate images from text descriptions. "
                                            "Images are saved to 'generated/images/' by default.",
                                            epilog="Example: openaicli image 'sunset over mountains' --model dall-e-3")
        image_parser.add_argument("prompt", nargs="?", help="Text description of the image to generate")
        image_parser.add_argument("--model", help="Model to use (e.g., dall-e-3, imagen-4.0)")
        image_parser.add_argument("--size", default="1024x1024", 
                                 help="Image size (e.g., 1024x1024, 1792x1024, 1024x1792)")
        image_parser.add_argument("--quality", default="standard", choices=["standard", "hd"],
                                 help="Image quality (standard or hd)")
        image_parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
        image_parser.add_argument("--output", "-o", help="Output directory (default: generated/images/)")

        # Video generation command
        video_parser = subparsers.add_parser("video", help="Generate videos from text prompts.",
                                            description="Generate videos from text descriptions. "
                                            "Videos are saved to 'generated/videos/' by default.",
                                            epilog="Example: openaicli video 'ocean waves crashing' --model sora-2")
        video_parser.add_argument("prompt", nargs="?", help="Text description of the video to generate")
        video_parser.add_argument("--model", help="Model to use (e.g., sora-2, veo-3.1)")
        video_parser.add_argument("--output", "-o", help="Output file path (default: generated/videos/video_TIMESTAMP.mp4)")

        # Text-to-speech command
        tts_parser = subparsers.add_parser("tts", help="Convert text to speech.",
                                          description="Convert text to speech audio. "
                                          "Audio files are saved to 'generated/audio/' by default.",
                                          epilog="Example: openaicli tts 'Hello world' --voice nova")
        tts_parser.add_argument("text", nargs="?", help="Text to convert to speech")
        tts_parser.add_argument("--model", default="tts-1", help="TTS model to use")
        tts_parser.add_argument("--voice", default="alloy",
                               choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                               help="Voice to use")
        tts_parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.25-4.0)")
        tts_parser.add_argument("--output", "-o", help="Output audio file path (default: generated/audio/speech_TIMESTAMP.mp3)")

        # List models command
        list_parser = subparsers.add_parser("models", help="List available models.")
        list_parser.add_argument("--type", help="Filter by type (chat, image, video, tts, etc.)")

        args = parser.parse_args()

        if args.command == "chat":
            # Only load config when actually starting a chat
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base, 
                                         self.config_manager.api_key, 
                                         self.console)
            self.temperature = args.temperature
            self.max_tokens = args.max_tokens
            self.start_chat_session(args.model, args.system, args.lang)
        
        elif args.command == "image":
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base,
                                         self.config_manager.api_key,
                                         self.console)
            self.handle_image_generation(args)
        
        elif args.command == "video":
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base,
                                         self.config_manager.api_key,
                                         self.console)
            self.handle_video_generation(args)
        
        elif args.command == "tts":
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base,
                                         self.config_manager.api_key,
                                         self.console)
            self.handle_text_to_speech(args)
        
        elif args.command == "models":
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base,
                                         self.config_manager.api_key,
                                         self.console)
            self.handle_list_models(args)
        
        elif args.command is None:
            # No command provided, show help
            parser.print_help()
        else:
            parser.print_help()

    def start_chat_session(self, model: Optional[str], system_prompt: Optional[str], language: str = "en") -> None:
        """Starts and manages the interactive chat session."""
        if not model:
            models = self.chat_client.list_models("/v1/chat/completions")
            if not models:
                self.console.print("[bold red]Could not fetch models. Please specify a model using the --model argument.[/bold red]")
                sys.exit(1)
            
            self.console.print("[bold]Available chat models:[/bold]")
            for i, m in enumerate(models):
                model_id = m.get("id", "unknown")
                owner = m.get("owned_by", "unknown")
                self.console.print(f"  [cyan]{i + 1}[/cyan]: {model_id} [dim]({owner})[/dim]")
            
            while True:
                try:
                    choice = input(f"Select a model (1-{len(models)}): ")
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(models):
                        model = models[model_index].get("id")
                        break
                    else:
                        self.console.print("[bold red]Invalid choice. Please try again.[/bold red]")
                except ValueError:
                    self.console.print("[bold red]Invalid input. Please enter a number.[/bold red]")

        history: List[Dict[str, str]] = []
        
        # Add language preference to system prompt if not already provided
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
            self.console.print(Panel(system_prompt, title="System Prompt", border_style="yellow"))
        elif language.lower() == "en":
            # Add implicit English preference for better UX
            lang_prompt = "Please respond in English."
            history.append({"role": "system", "content": lang_prompt})

        # Display welcome message
        self.clear_screen()
        self.console.print(WELCOME_MSG)
        
        # Initialize seed
        if self.seed is None:
            self.seed = random.randint(1, 10000)
        
        while True:
            try:
                # Check history size and warn if getting large
                if len(history) > self.max_history_messages:
                    self.console.print(f"[yellow]Warning: Conversation history has {len(history)} messages. Consider using /clear to start fresh.[/yellow]")
                
                prompt = self.get_user_input()
                
                if not prompt or prompt.isspace():
                    self.console.print("[dim]Empty message, please try again.[/dim]")
                    continue
                
                # Handle both / and : prefixed commands
                if prompt.lower().startswith('/') or prompt.lower().startswith(':'):
                    should_continue = self.handle_command(prompt, history, model)
                    if not should_continue:
                        break
                    continue
                
                history.append({"role": "user", "content": prompt})
                
                try:
                    history, assistant_response = self.chat_client.stream_chat(
                        history, model, self.temperature, self.max_tokens
                    )
                    # Response already printed during streaming, just add panel around it
                    self.console.print()  # Extra newline for spacing
                except Exception as e:
                    self.console.print(f"[bold red]Failed to get response: {e}[/bold red]")
                    # Remove the user message from history since we didn't get a response
                    history.pop()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Use /exit to quit or continue chatting.[/yellow]")
                continue
            except EOFError:
                break
        
        self.console.print("\n[bold]Chat session ended.[/bold]")

    def get_user_input(self) -> str:
        """Gets user input with support for both single-line and multiline."""
        print()  # Add spacing before prompt
        self.console.print("[bold white]┌─[/bold white][bold blue]You[/bold blue][bold white]:[/bold white]")
        self.console.print("[bold white]└─>[/bold white] ", end="")
        
        try:
            first_line = input()
            
            # Check if user wants multiline input (ends with backslash or is empty and they continue)
            if first_line.endswith("\\"):
                # Multiline mode triggered by backslash
                self.console.print("[dim]   (Multiline mode - type empty line to finish)[/dim]")
                lines = [first_line[:-1]]  # Remove trailing backslash
                while True:
                    self.console.print("[bold white]   │[/bold white] ", end="")
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                return "\n".join(lines)
            else:
                return first_line
        except EOFError:
            # Ctrl+D pressed - multiline input
            self.console.print("\n[dim]   (Multiline mode - type Ctrl+D again to finish)[/dim]")
            lines = []
            try:
                while True:
                    self.console.print("[bold white]   │[/bold white] ", end="")
                    lines.append(input())
            except EOFError:
                pass
            return "\n".join(lines)

    def handle_command(self, prompt: str, history: List[Dict[str, str]], model: str) -> bool:
        """Handles special commands. Returns False if should exit, True to continue."""
        # Remove prefix (/ or :) and parse
        prompt = prompt[1:].strip()
        parts = prompt.split()
        if not parts:
            return True
            
        command = parts[0].lower()
        args = parts[1:]
        
        # Save command
        if command in ["save"]:
            if args:
                self.save_history(history, args[0])
            else:
                self.console.print("[bold red]Usage: :save <filename>[/bold red]")
                
        # Load command
        elif command in ["load"]:
            if args:
                loaded = self.load_history(args[0])
                if loaded:
                    history.clear()
                    history.extend(loaded)
                    self.console.print(f"[green]Loaded {len(loaded)} messages from history[/green]")
            else:
                self.console.print("[bold red]Usage: :load <filename>[/bold red]")
        
        # Clear screen command
        elif command in ["clear", "cl"]:
            self.clear_screen()
            self.console.print(WELCOME_MSG)
                
        # Clear history command
        elif command in ["clear-history", "clh", "clear-his"]:
            history.clear()
            self.console.print("[bold yellow]Conversation history cleared.[/bold yellow]")
            
        # History command
        elif command in ["history", "his"]:
            if not history:
                self.console.print("[dim]No conversation history yet.[/dim]")
            else:
                self.console.print(f"[bold]Conversation History ({len(history)} messages):[/bold]")
                for i, msg in enumerate(history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    preview = content[:50] + "..." if len(content) > 50 else content
                    self.console.print(f"  {i+1}. [{role}]: {preview}")
        
        # Seed command
        elif command in ["seed"]:
            if args:
                try:
                    new_seed = int(args[0])
                    self.seed = new_seed
                    random.seed(self.seed)
                    self.console.print(f"[green]Random seed set to {self.seed}[/green]")
                except ValueError:
                    self.console.print(f"[bold red]Invalid seed: {args[0]} is not a valid number[/bold red]")
            else:
                self.console.print(f"[cyan]Current random seed: {self.seed}[/cyan]")
        
        # Configuration command
        elif command in ["conf", "config"]:
            if args:
                # Parse key=value pairs
                for arg in args:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == "temperature":
                            try:
                                new_temp = float(value)
                                if 0.0 <= new_temp <= 2.0:
                                    self.temperature = new_temp
                                    self.console.print(f"[green]Temperature set to {self.temperature}[/green]")
                                else:
                                    self.console.print("[bold red]Temperature must be between 0.0 and 2.0[/bold red]")
                            except ValueError:
                                self.console.print(f"[bold red]Invalid temperature value: {value}[/bold red]")
                        
                        elif key == "max_tokens":
                            try:
                                new_tokens = int(value)
                                if new_tokens > 0:
                                    self.max_tokens = new_tokens
                                    self.console.print(f"[green]Max tokens set to {self.max_tokens}[/green]")
                                else:
                                    self.console.print("[bold red]Max tokens must be positive[/bold red]")
                            except ValueError:
                                self.console.print(f"[bold red]Invalid max_tokens value: {value}[/bold red]")
                        
                        else:
                            self.console.print(f"[bold red]Unknown configuration key: {key}[/bold red]")
                            self.console.print("[dim]Available keys: temperature, max_tokens[/dim]")
                    else:
                        self.console.print(f"[bold red]Invalid format: {arg}. Use key=value[/bold red]")
            else:
                # Show current configuration
                self.console.print("[bold]Current Configuration:[/bold]")
                self.console.print(f"  Temperature: [cyan]{self.temperature}[/cyan]")
                max_tok = self.max_tokens if self.max_tokens else "unlimited"
                self.console.print(f"  Max Tokens: [cyan]{max_tok}[/cyan]")
                self.console.print(f"  Random Seed: [cyan]{self.seed}[/cyan]")
                self.console.print(f"  Model: [cyan]{model}[/cyan]")
        
        # Reset configuration command
        elif command in ["reset-conf", "reset-config"]:
            self.temperature = 0.7
            self.max_tokens = None
            self.console.print("[green]Configuration reset to defaults[/green]")
            self.console.print(f"  Temperature: {self.temperature}")
            self.console.print(f"  Max Tokens: unlimited")
                    
        # Temperature command (legacy support)
        elif command in ["temp"]:
            if args:
                try:
                    new_temp = float(args[0])
                    if 0.0 <= new_temp <= 2.0:
                        self.temperature = new_temp
                        self.console.print(f"[green]Temperature set to {self.temperature}[/green]")
                    else:
                        self.console.print("[bold red]Temperature must be between 0.0 and 2.0[/bold red]")
                except ValueError:
                    self.console.print("[bold red]Invalid temperature value[/bold red]")
            else:
                self.console.print(f"[cyan]Current temperature: {self.temperature}[/cyan]")
                self.console.print("[dim]Usage: :temp <value> (0.0-2.0)[/dim]")
                
        # Tokens command (legacy support)
        elif command in ["tokens"]:
            if args:
                try:
                    new_tokens = int(args[0])
                    if new_tokens > 0:
                        self.max_tokens = new_tokens
                        self.console.print(f"[green]Max tokens set to {self.max_tokens}[/green]")
                    else:
                        self.console.print("[bold red]Max tokens must be positive[/bold red]")
                except ValueError:
                    self.console.print("[bold red]Invalid token value[/bold red]")
            else:
                current = self.max_tokens if self.max_tokens else "unlimited"
                self.console.print(f"[cyan]Current max tokens: {current}[/cyan]")
                self.console.print("[dim]Usage: :tokens <number>[/dim]")
                
        # Model command
        elif command in ["model"]:
            self.console.print(f"[cyan]Current model: {model}[/cyan]")
            self.console.print("[dim]Note: To change model, restart the chat session[/dim]")
        
        # YOLO mode command
        elif command in ["yolo"]:
            self.yolo_mode = not self.yolo_mode
            if self.yolo_mode:
                self.console.print("[bold green]┌─ YOLO Mode: ENABLED[/bold green]")
                self.console.print("[green]└─>[/green] [dim]AI can now analyze your source code. Use :analyze to scan the project.[/dim]")
                # Initialize with current directory if not set
                if not self.current_project:
                    self.current_project = os.getcwd()
                    self.code_analyzer = CodeAnalyzer(self.current_project, self.console)
                    self.console.print(f"[dim]   Current project: {self.current_project}[/dim]")
            else:
                self.console.print("[bold red]┌─ YOLO Mode: DISABLED[/bold red]")
                self.console.print("[red]└─>[/red] [dim]Code analysis features are now disabled.[/dim]")
        
        # Analyze project command
        elif command in ["analyze", "scan"]:
            if not self.yolo_mode:
                self.console.print("[bold yellow]YOLO mode is not enabled. Use :yolo to enable it first.[/bold yellow]")
            else:
                if not self.code_analyzer:
                    self.code_analyzer = CodeAnalyzer(self.current_project or os.getcwd(), self.console)
                
                stats = self.code_analyzer.scan_project()
                
                if stats:
                    # Enhanced display with project type and key files
                    project_type_display = f"Type: [magenta]{stats.get('project_type', 'unknown').title()}[/magenta]" if stats.get('project_type') else ""
                    
                    panel_content = (
                        f"[bold]Project Analysis[/bold]\n\n"
                        f"Location: [cyan]{stats['project_path']}[/cyan]\n"
                        f"{project_type_display}\n" if project_type_display else ""
                        f"Files: [green]{stats['total_files']}[/green]\n"
                        f"Lines: [green]{stats['total_lines']:,}[/green]\n\n"
                        f"[bold]Top Languages:[/bold]\n"
                    )
                    
                    # Show top 5 file types
                    top_types = sorted(stats['file_stats'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for ext, lines in top_types:
                        panel_content += f"  {ext}: {lines:,} lines\n"
                    
                    # Show key files if found
                    if stats.get('key_files'):
                        panel_content += f"\n[bold]Key Files Found:[/bold]\n"
                        for kf in stats['key_files'][:5]:
                            panel_content += f"  - {kf}\n"
                    
                    self.console.print(Panel(
                        panel_content,
                        title="[bold white]Workspace Analysis[/bold white]",
                        border_style="blue",
                        padding=(1, 2)
                    ))
                    
                    # Show tree
                    self.console.print("\n[bold]Project Structure:[/bold]")
                    tree = self.code_analyzer.get_project_tree()
                    self.console.print(tree)
                    
                    # Generate comprehensive summary with code samples
                    summary = self.code_analyzer.generate_summary()
                    
                    # Add sample code from important files
                    self.console.print("\n[cyan]Reading key files for AI context...[/cyan]")
                    code_samples = []
                    
                    # Prioritize important files (max 10 files, 100 lines each)
                    important_patterns = [
                        'README.md', 'package.json', 'tsconfig.json', 'vite.config',
                        'main.', 'index.', 'App.', 'server.', 'api.', 'config.',
                        'schema.', 'routes.', 'middleware.'
                    ]
                    
                    sampled_files = []
                    for pattern in important_patterns:
                        matching = [f for f in self.code_analyzer.files if pattern in f and f not in sampled_files]
                        if matching:
                            sampled_files.append(matching[0])
                        if len(sampled_files) >= 10:
                            break
                    
                    # Add any remaining important files
                    if len(sampled_files) < 10:
                        for file in self.code_analyzer.files[:20]:
                            if file not in sampled_files:
                                sampled_files.append(file)
                            if len(sampled_files) >= 10:
                                break
                    
                    for file in sampled_files:
                        content = self.code_analyzer.read_file_content(file, max_lines=100)
                        if content and not content.startswith("Error"):
                            code_samples.append(f"\n--- {file} ---\n{content}")
                    
                    # Combine summary with code samples
                    full_context = f"{summary}\n\n{'=' * 60}\nCODE SAMPLES\n{'=' * 60}\n" + "\n".join(code_samples)
                    
                    # Add to chat context
                    history.append({
                        "role": "system",
                        "content": f"Project analysis complete. You now have full context of this codebase:\n\n{full_context}\n\nYou can now answer questions about this code, suggest improvements, find bugs, explain architecture, etc."
                    })
                    self.console.print(f"\n[green]> Project context added ({len(sampled_files)} files analyzed)[/green]")
                    self.console.print("[dim]You can now ask questions about the codebase![/dim]")
        
        # Read specific file command
        elif command in ["read", "cat", "show"]:
            if not self.yolo_mode:
                self.console.print("[bold yellow]YOLO mode is not enabled. Use :yolo to enable it first.[/bold yellow]")
            elif not args:
                self.console.print("[bold red]Usage: :read <filename>[/bold red]")
                self.console.print("[dim]Example: :read src/App.tsx[/dim]")
            else:
                if not self.code_analyzer:
                    self.code_analyzer = CodeAnalyzer(self.current_project or os.getcwd(), self.console)
                    self.code_analyzer.scan_project()
                
                filename = ' '.join(args)
                
                # Try to find the file (support partial matches)
                matching_files = [f for f in self.code_analyzer.files if filename in f]
                
                if not matching_files:
                    self.console.print(f"[bold red]File not found: {filename}[/bold red]")
                    self.console.print("[dim]Use :files to list all files[/dim]")
                elif len(matching_files) > 1:
                    self.console.print(f"[bold yellow]Multiple files match '{filename}':[/bold yellow]")
                    for i, f in enumerate(matching_files[:10], 1):
                        self.console.print(f"  {i}. {f}")
                    self.console.print("[dim]Please be more specific[/dim]")
                else:
                    file_to_read = matching_files[0]
                    content = self.code_analyzer.read_file_content(file_to_read, max_lines=500)
                    
                    if content:
                        self.console.print(Panel(
                            f"[dim]{content}[/dim]",
                            title=f"[bold white]File: {file_to_read}[/bold white]",
                            border_style="blue",
                            padding=(1, 2)
                        ))
                        
                        # Add to context
                        history.append({
                            "role": "system",
                            "content": f"File: {file_to_read}\n\n{content}"
                        })
                        self.console.print(f"[green]> File content added to context[/green]")
                    else:
                        self.console.print(f"[bold red]Could not read file: {file_to_read}[/bold red]")
        
        # List files command
        elif command in ["files", "ls"]:
            if not self.yolo_mode:
                self.console.print("[bold yellow]YOLO mode is not enabled. Use :yolo to enable it first.[/bold yellow]")
            else:
                if not self.code_analyzer:
                    self.code_analyzer = CodeAnalyzer(self.current_project or os.getcwd(), self.console)
                    self.code_analyzer.scan_project()
                
                if args:
                    # Filter files by pattern
                    pattern = ' '.join(args).lower()
                    filtered = [f for f in self.code_analyzer.files if pattern in f.lower()]
                    self.console.print(f"[bold]Files matching '{pattern}':[/bold] ({len(filtered)} files)")
                    for f in filtered[:50]:
                        self.console.print(f"  [green]{f}[/green]")
                    if len(filtered) > 50:
                        self.console.print(f"[dim]  ... and {len(filtered) - 50} more[/dim]")
                else:
                    # List all files
                    self.console.print(f"[bold]All files:[/bold] ({len(self.code_analyzer.files)} files)")
                    for f in self.code_analyzer.files[:50]:
                        ext = os.path.splitext(f)[1]
                        if ext in ['.tsx', '.ts', '.jsx', '.js', '.py']:
                            self.console.print(f"  [green]{f}[/green]")
                        else:
                            self.console.print(f"  [cyan]{f}[/cyan]")
                    if len(self.code_analyzer.files) > 50:
                        self.console.print(f"[dim]  ... and {len(self.code_analyzer.files) - 50} more[/dim]")
                    self.console.print("\n[dim]Use :files <pattern> to filter[/dim]")
        
        # Project/CD command
        elif command in ["project", "cd"]:
            if args:
                new_path = ' '.join(args)
                # Expand ~ and resolve path
                new_path = os.path.expanduser(new_path)
                new_path = os.path.abspath(new_path)
                
                if os.path.exists(new_path) and os.path.isdir(new_path):
                    self.current_project = new_path
                    # CRITICAL FIX: Reset code_analyzer to None instead of creating new instance
                    # This ensures :analyze will scan the NEW project, not the old one
                    self.code_analyzer = None
                    self.config_manager.add_recent_project(self.current_project)
                    self.console.print(f"[green]> Switched to project: {self.current_project}[/green]")
                    
                    if self.yolo_mode:
                        self.console.print("[dim]Use :analyze to scan this project[/dim]")
                else:
                    self.console.print(f"[bold red]Error: Directory not found: {new_path}[/bold red]")
            else:
                # Show current project and recent projects
                if self.current_project:
                    self.console.print(f"[cyan]Current project: {self.current_project}[/cyan]")
                else:
                    self.console.print(f"[cyan]Current directory: {os.getcwd()}[/cyan]")
                
                if self.config_manager.recent_projects:
                    self.console.print("\n[bold]Recent projects:[/bold]")
                    for i, proj in enumerate(reversed(self.config_manager.recent_projects[-5:]), 1):
                        self.console.print(f"  {i}. {proj}")
                    self.console.print("\n[dim]Use :project <path> to switch[/dim]")
            
        # Help command
        elif command in ["help", "h"]:
            self.console.print(HELP_MSG)
            
        # Exit commands
        elif command in ["exit", "quit", "q"]:
            return False
            
        else:
            self.console.print(f"[bold red]Unknown command: :{command}[/bold red]")
            self.console.print("[dim]Type :help for available commands[/dim]")
        
        return True

    def handle_image_generation(self, args) -> None:
        """Handles image generation command."""
        import base64
        from datetime import datetime
        
        # Get prompt
        prompt = args.prompt
        if not prompt:
            self.console.print("[bold cyan]Enter your image prompt:[/bold cyan]")
            prompt = input("> ")
        
        if not prompt or prompt.isspace():
            self.console.print("[bold red]Error: Prompt cannot be empty[/bold red]")
            return
        
        # Select model if not specified
        model = args.model
        if not model:
            models = self.chat_client.list_models("/v1/images/generations")
            if models:
                self.console.print("[bold]Available image models:[/bold]")
                for i, m in enumerate(models):
                    model_id = m.get("id", "unknown")
                    owner = m.get("owned_by", "unknown")
                    self.console.print(f"  [cyan]{i + 1}[/cyan]: {model_id} [dim]({owner})[/dim]")
                
                try:
                    choice = input(f"Select a model (1-{len(models)}) or press Enter for default: ")
                    if choice.strip():
                        model_index = int(choice) - 1
                        if 0 <= model_index < len(models):
                            model = models[model_index].get("id")
                except:
                    pass
            
            if not model:
                model = "dall-e-3"  # Default
        
        # Generate images
        urls = self.chat_client.generate_image(prompt, model, args.size, args.quality, args.n)
        
        if not urls:
            return
        
        # Create organized output directory
        if args.output:
            output_dir = args.output
        else:
            output_dir = os.path.join("generated", "images")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, url_or_b64 in enumerate(urls):
            filename = f"image_{timestamp}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            
            try:
                if url_or_b64.startswith("http"):
                    # Download from URL
                    self.console.print(f"[cyan]Downloading image {i+1}/{len(urls)}...[/cyan]")
                    img_response = requests.get(url_or_b64, timeout=30)
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                else:
                    # Save from base64
                    self.console.print(f"[cyan]Saving image {i+1}/{len(urls)}...[/cyan]")
                    img_data = base64.b64decode(url_or_b64)
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                
                self.console.print(f"[green]> Saved to {filepath}[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error saving image: {e}[/bold red]")

    def handle_video_generation(self, args) -> None:
        """Handles video generation command."""
        import base64
        from datetime import datetime
        
        # Get prompt
        prompt = args.prompt
        if not prompt:
            self.console.print("[bold cyan]Enter your video prompt:[/bold cyan]")
            prompt = input("> ")
        
        if not prompt or prompt.isspace():
            self.console.print("[bold red]Error: Prompt cannot be empty[/bold red]")
            return
        
        # Select model if not specified
        model = args.model
        if not model:
            models = self.chat_client.list_models("/v1/videos")
            if models:
                self.console.print("[bold]Available video models:[/bold]")
                for i, m in enumerate(models):
                    model_id = m.get("id", "unknown")
                    owner = m.get("owned_by", "unknown")
                    self.console.print(f"  [cyan]{i + 1}[/cyan]: {model_id} [dim]({owner})[/dim]")
                
                try:
                    choice = input(f"Select a model (1-{len(models)}) or press Enter for default: ")
                    if choice.strip():
                        model_index = int(choice) - 1
                        if 0 <= model_index < len(models):
                            model = models[model_index].get("id")
                except:
                    pass
            
            if not model:
                model = "sora-2"  # Default
        
        # Generate video
        url_or_b64 = self.chat_client.generate_video(prompt, model)
        
        if not url_or_b64:
            return
        
        # Create organized output directory and file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output:
            filename = args.output
        else:
            output_dir = os.path.join("generated", "videos")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"video_{timestamp}.mp4")
        
        try:
            if url_or_b64.startswith("http"):
                # Download from URL
                self.console.print("[cyan]Downloading video...[/cyan]")
                vid_response = requests.get(url_or_b64, timeout=300, stream=True)
                with open(filename, 'wb') as f:
                    for chunk in vid_response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                # Save from base64
                self.console.print("[cyan]Saving video...[/cyan]")
                vid_data = base64.b64decode(url_or_b64)
                with open(filename, 'wb') as f:
                    f.write(vid_data)
            
            self.console.print(f"[green]> Video saved to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Error saving video: {e}[/bold red]")

    def handle_text_to_speech(self, args) -> None:
        """Handles text-to-speech command."""
        from datetime import datetime
        
        # Get text
        text = args.text
        if not text:
            self.console.print("[bold cyan]Enter text to convert to speech:[/bold cyan]")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            text = "\n".join(lines)
        
        if not text or text.isspace():
            self.console.print("[bold red]Error: Text cannot be empty[/bold red]")
            return
        
        # Generate speech
        audio_data = self.chat_client.text_to_speech(text, args.model, args.voice, args.speed)
        
        if not audio_data:
            return
        
        # Create organized output directory and file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output:
            filename = args.output
        else:
            output_dir = os.path.join("generated", "audio")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"speech_{timestamp}.mp3")
        
        try:
            with open(filename, 'wb') as f:
                f.write(audio_data)
            self.console.print(f"[green]> Audio saved to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Error saving audio: {e}[/bold red]")

    def handle_list_models(self, args) -> None:
        """Handles listing available models."""
        # Map type shortcuts to full paths
        type_map = {
            "chat": "/v1/chat/completions",
            "image": "/v1/images/generations",
            "video": "/v1/videos",
            "tts": "/v1/audio/speech",
            "transcribe": "/v1/audio/transcriptions",
            "embed": "/v1/embeddings"
        }
        
        model_type = None
        if args.type:
            model_type = type_map.get(args.type.lower(), args.type)
        
        models = self.chat_client.list_models(model_type)
        
        if not models:
            self.console.print("[yellow]No models found[/yellow]")
            return
        
        # Group models by owner
        from collections import defaultdict
        by_owner = defaultdict(list)
        for model in models:
            owner = model.get("owned_by", "unknown")
            by_owner[owner].append(model)
        
        self.console.print(f"\n[bold]Found {len(models)} models[/bold]\n")
        
        for owner, owner_models in sorted(by_owner.items()):
            self.console.print(f"[bold cyan]{owner.upper()}[/bold cyan] ({len(owner_models)} models)")
            for model in sorted(owner_models, key=lambda x: x.get("id", "")):
                model_id = model.get("id", "unknown")
                types = model.get("type", [])
                if types:
                    type_str = ", ".join([t.split("/")[-1] for t in types[:2]])
                    if len(types) > 2:
                        type_str += f", +{len(types)-2} more"
                    self.console.print(f"  - {model_id} [dim]({type_str})[/dim]")
                else:
                    self.console.print(f"  - {model_id}")
            self.console.print()

    def save_history(self, history: List[Dict[str, str]], filename: str) -> None:
        """Saves conversation history to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            self.console.print(f"[green]Conversation saved to {filename}[/green]")
            self.console.print(f"[dim]Saved {len(history)} messages[/dim]")
        except IOError as e:
            self.console.print(f"[bold red]Error saving history: {e}[/bold red]")

    def load_history(self, filename: str) -> List[Dict[str, str]]:
        """Loads conversation history from a file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
                
                # Validate loaded data
                if not isinstance(history, list):
                    self.console.print("[bold red]Error: Invalid history file format (not a list)[/bold red]")
                    return []
                
                # Validate each message has required fields
                valid_history = []
                for msg in history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        valid_history.append(msg)
                    else:
                        self.console.print(f"[yellow]Warning: Skipping invalid message in history[/yellow]")
                
                if valid_history:
                    self.console.print(f"[green]Conversation loaded from {filename}[/green]")
                    return valid_history
                else:
                    self.console.print("[bold red]Error: No valid messages found in history file[/bold red]")
                    return []
                    
        except FileNotFoundError:
            self.console.print(f"[bold red]Error: File '{filename}' not found[/bold red]")
            return []
        except json.JSONDecodeError as e:
            self.console.print(f"[bold red]Error: Invalid JSON in history file: {e}[/bold red]")
            return []
        except IOError as e:
            self.console.print(f"[bold red]Error loading history: {e}[/bold red]")
            return []

def main() -> None:
    """Main function for the CLI."""
    try:
        cli = CLI()
        cli.run()
    except ImportError as e:
        print(f"Error: Missing required library - {e}")
        print("Please install required libraries: pip install rich requests")
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

