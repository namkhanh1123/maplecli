import argparse
import os
import requests
import sys
import json
import stat
import platform
import random
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

WELCOME_MSG = """\
[bold cyan]Welcome to MapleCLI - OpenAI Compatible Chat Interface[/bold cyan]

Type your message to chat with the AI.
Type [yellow]:help[/yellow] or [yellow]:h[/yellow] to see available commands.

[dim]Press Ctrl+C to interrupt, Ctrl+D for multiline input.[/dim]
"""

HELP_MSG = """\
[bold]Available Commands:[/bold]
  [yellow]:help / :h[/yellow]            Show this help message
  [yellow]:exit / :quit / :q[/yellow]    Exit the chat session
  [yellow]:clear / :cl[/yellow]          Clear the screen
  [yellow]:clear-history / :clh[/yellow] Clear conversation history
  [yellow]:history / :his[/yellow]       Show conversation history
  [yellow]:seed[/yellow]                 Show current random seed
  [yellow]:seed <N>[/yellow]             Set random seed to <N>
  [yellow]:conf[/yellow]                 Show current configuration
  [yellow]:conf <key>=<value>[/yellow]   Change configuration (e.g., :conf temperature=0.8)
  [yellow]:reset-conf[/yellow]           Reset configuration to defaults
  [yellow]:save <file>[/yellow]          Save conversation to file
  [yellow]:load <file>[/yellow]          Load conversation from file
  [yellow]:model[/yellow]                Show current model
  
[bold]Configuration Keys:[/bold]
  - temperature (0.0-2.0)
  - max_tokens (integer)
  
[bold]Input Tips:[/bold]
  - End line with \\ for multiline input
  - Press Ctrl+D to enter multiline mode
"""

class ConfigManager:
    """Manages the configuration for the CLI."""
    def __init__(self, console: Console):
        self.console = console
        self.config_dir = os.path.expanduser("~/.config/openaicli")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.api_base: Optional[str] = None
        self.api_key: Optional[str] = None

    def load_config(self) -> None:
        """Loads configuration from file or environment variables."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                try:
                    config = json.load(f)
                    self.api_base = config.get("OPENAI_API_BASE")
                    self.api_key = config.get("OPENAI_API_KEY")
                except json.JSONDecodeError as e:
                    self.console.print(f"[bold yellow]Warning: Config file corrupted ({e}). Using environment variables or prompting.[/bold yellow]")

        if not self.api_base:
            self.api_base = os.environ.get("OPENAI_API_BASE")
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_base:
            self.console.print("[bold red]OpenAI API base URL not found.[/bold red]")
            self.api_base = input("Please enter your OpenAI API base URL: ")
            self.api_key = input("Please enter your OpenAI API key (optional): ")
            self.save_config()

    def save_config(self) -> None:
        """Saves API configuration to the config file with proper permissions."""
        os.makedirs(self.config_dir, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump({"OPENAI_API_BASE": self.api_base, "OPENAI_API_KEY": self.api_key}, f)
        
        # Set restrictive permissions on config file (owner read/write only)
        try:
            os.chmod(self.config_file, stat.S_IRUSR | stat.S_IWUSR)
        except (OSError, AttributeError):
            # Windows or permission error - continue anyway
            pass
        
        self.console.print(f"Configuration saved to [green]{self.config_file}[/green]")

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
        
        # Show a subtle status indicator
        self.console.print("[dim cyan]◌ Thinking...[/dim cyan]", end="")
        
        for attempt in range(self.max_retries):
            try:
                with requests.post(f"{self.api_base}/chat/completions", 
                                 headers=headers, json=data, stream=True, 
                                 timeout=self.timeout) as response:
                    if response.status_code == 200:
                        # Clear the "Thinking..." message completely
                        self.console.print("\r" + " " * 50 + "\r", end="")
                        
                        # Print the assistant label before streaming
                        self.console.print("[bold green]Assistant:[/bold green] ", end="")
                        
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
        self.console.print("[bold blue]You:[/bold blue] ", end="")
        
        try:
            first_line = input()
            
            # Check if user wants multiline input (ends with backslash or is empty and they continue)
            if first_line.endswith("\\"):
                # Multiline mode triggered by backslash
                self.console.print("[dim](Multiline mode - type empty line to finish)[/dim]")
                lines = [first_line[:-1]]  # Remove trailing backslash
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                return "\n".join(lines)
            else:
                return first_line
        except EOFError:
            # Ctrl+D pressed - multiline input
            self.console.print("\n[dim](Multiline mode - type Ctrl+D again to finish)[/dim]")
            lines = []
            try:
                while True:
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
                
                self.console.print(f"[green]✓ Saved to {filepath}[/green]")
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
            
            self.console.print(f"[green]✓ Video saved to {filename}[/green]")
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
            self.console.print(f"[green]✓ Audio saved to {filename}[/green]")
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
                    self.console.print(f"  • {model_id} [dim]({type_str})[/dim]")
                else:
                    self.console.print(f"  • {model_id}")
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

