"""
MapleCLI Main CLI Class
"""
import os
import sys
import json
import platform
import random
import argparse
from typing import List, Dict, Optional

from rich.console import Console
from rich.panel import Panel

from config_manager import ConfigManager
from chat_client import ChatClient
from code_analyzer import CodeAnalyzer
from logger import maple_logger

# Import intelligent context components with graceful fallback
try:
    from context_engine import ContextEngine
    CONTEXT_ENGINE_AVAILABLE = True
except ImportError:
    CONTEXT_ENGINE_AVAILABLE = False

try:
    from symbol_resolver import SymbolResolver
    SYMBOL_RESOLVER_AVAILABLE = True
except ImportError:
    SYMBOL_RESOLVER_AVAILABLE = False

try:
    from query_analyzer import QueryAnalyzer
    QUERY_ANALYZER_AVAILABLE = True
except ImportError:
    QUERY_ANALYZER_AVAILABLE = False

WELCOME_MSG = """[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]      Welcome to MapleCLI Chat!       [bold cyan]║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]

[dim]Type your message and press Enter to chat.[/dim]
[dim]Type :help for available commands or :exit to quit.[/dim]

[yellow]💡 Tip: If a model responds in another language, just ask it to respond in English.[/yellow]
"""

HELP_MSG = """[bold]Available Commands:[/bold]
  :exit, :q          - Exit the chat session
  :yolo              - Toggle YOLO mode (intelligent context)
  :analyze           - Analyze project for intelligent context
  :clear             - Clear chat history
  :temp <value>      - Set temperature (0.0-2.0)
  :tokens <value>    - Set max tokens
  :system <prompt>   - Set system prompt
  :help              - Show this help message
"""

class CLI:
    """Handles the command-line interface and chat loop."""
    def __init__(self):
        self.console = Console()
        self.config_manager = ConfigManager(self.console)
        self.chat_client: Optional[ChatClient] = None
        self.temperature = 0.7
        self.max_tokens: Optional[int] = None
        self.yolo_mode = False
        self.current_project: Optional[str] = None
        self.code_analyzer: Optional[CodeAnalyzer] = None
        self.context_engine: Optional['ContextEngine'] = None
        self.symbol_resolver: Optional['SymbolResolver'] = None
        self.query_analyzer: Optional['QueryAnalyzer'] = None

    def run(self) -> None:
        """Runs the main CLI logic."""
        parser = argparse.ArgumentParser(description="A CLI for OpenAI-compatible APIs.")
        subparsers = parser.add_subparsers(dest="command")

        chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
        chat_parser.add_argument("--model", help="The model to use for the chat.")
        chat_parser.add_argument("--system", help="Set a system prompt for the chat session.")
        
        image_parser = subparsers.add_parser("image", help="Generate images from text prompts.")
        image_parser.add_argument("prompt", nargs="?", help="Text description of the image to generate")

        # ... (other parsers for video, tts, models)

        args = parser.parse_args()

        if args.command == "chat":
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base, self.config_manager.api_key, self.console)
            self.start_chat_session(args.model, args.system)
        elif args.command == "image":
            self.config_manager.load_config()
            self.chat_client = ChatClient(self.config_manager.api_base, self.config_manager.api_key, self.console)
            self.handle_image_generation(args)
        else:
            parser.print_help()

    def start_chat_session(self, model: Optional[str], system_prompt: Optional[str]) -> None:
        """Starts and manages the interactive chat session."""
        if not model:
            # Get available models and let user choose
            models = self.chat_client.list_models()
            if not models:
                self.console.print("[bold red]Error: Could not fetch models. Using default 'gpt-3.5-turbo'[/bold red]")
                model = "gpt-3.5-turbo"
            elif len(models) == 1:
                model = models[0]["id"]
                self.console.print(f"[cyan]Using model: {model}[/cyan]")
            else:
                self.console.print("[bold cyan]Available models:[/bold cyan]")
                for i, m in enumerate(models, 1):
                    self.console.print(f"  {i}. {m['id']}")
                
                # Try to find a good default
                default_model = next((m['id'] for m in models if 'gpt-4o-mini' in m['id'].lower()), 
                                   next((m['id'] for m in models if 'gpt-4o' in m['id'].lower()), models[0]['id']))
                
                try:
                    choice = input(f"\nSelect a model (number or name, or Enter for {default_model}): ").strip()
                    if not choice:
                        model = default_model
                    elif choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(models):
                            model = models[idx]["id"]
                        else:
                            self.console.print("[yellow]Invalid choice. Using default.[/yellow]")
                            model = default_model
                    else:
                        model = choice
                except (EOFError, KeyboardInterrupt):
                    self.console.print(f"\n[yellow]Using default: {default_model}[/yellow]")
                    model = default_model
        
        self.console.print(f"[bold green]Starting chat with model: {model}[/bold green]")

        history: List[Dict[str, str]] = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})

        self.console.print(WELCOME_MSG)
        
        while True:
            try:
                prompt = self.get_user_input()
                if not prompt: continue

                if prompt.lower().startswith(':'):
                    if not self.handle_command(prompt, history, model):
                        break
                    continue
                
                enhanced_prompt = self.enhance_prompt_with_context(prompt)
                history.append({"role": "user", "content": enhanced_prompt})
                
                history, _ = self.chat_client.stream_chat(history, model, self.temperature, self.max_tokens)

            except (KeyboardInterrupt, EOFError):
                break
        
        self.console.print("\n[bold]Chat session ended.[/bold]")

    def enhance_prompt_with_context(self, prompt: str) -> str:
        """Enhances the user prompt with relevant code context if in YOLO mode."""
        if not self.yolo_mode or not self.code_analyzer:
            return prompt

        if self.query_analyzer and self.symbol_resolver and self.context_engine:
            analysis = self.query_analyzer.analyze(prompt)
            
            if self.query_analyzer.should_use_symbol_search(analysis) and analysis['entities']:
                symbol_name = analysis['entities'][0]
                definitions = self.symbol_resolver.find_definition(symbol_name)
                if definitions:
                    context = "\n".join([f"[{d['file']}:{d['line']}]\n{self.code_analyzer.read_file_content(d['file'])}" for d in definitions[:3]])
                    return f"{prompt}\n\n[DEFINITIONS FOUND]:\n{context}"
            
            elif self.query_analyzer.should_use_semantic_search(analysis):
                relevant_chunks = self.context_engine.search(prompt, top_k=5)
                if relevant_chunks:
                    context = "\n".join([f"[{c['filepath']}:{c['start_line']}]\n{c['content']}" for c in relevant_chunks])
                    return f"{prompt}\n\n[RELEVANT CODE]:\n{context}"
        
        return prompt

    def get_user_input(self) -> str:
        """Gets user input with support for multiline."""
        try:
            self.console.print("\n[bold cyan]You:[/bold cyan] ", end="")
            user_input = input().strip()
            
            # Check for multiline input (ending with \)
            while user_input.endswith('\\'):
                user_input = user_input[:-1] + '\n'
                self.console.print("[bold cyan]...[/bold cyan] ", end="")
                user_input += input().strip()
            
            return user_input
        except (EOFError, KeyboardInterrupt):
            return ":exit"

    def handle_command(self, prompt: str, history: List[Dict[str, str]], model: str) -> bool:
        """Handles special commands."""
        command, *args = prompt[1:].strip().split()
        
        if command in ["exit", "q"]:
            return False
        elif command == "yolo":
            self.yolo_mode = not self.yolo_mode
            self.console.print(f"[green]YOLO mode {'ENABLED' if self.yolo_mode else 'DISABLED'}[/green]")
            if self.yolo_mode and not self.current_project:
                self.current_project = os.getcwd()
        elif command == "analyze":
            if self.yolo_mode:
                self.code_analyzer = CodeAnalyzer(self.current_project, self.console)
                self.code_analyzer.scan_project()
                if CONTEXT_ENGINE_AVAILABLE:
                    self.context_engine = ContextEngine(self.current_project, self.console)
                    self.context_engine.build_index(self.code_analyzer)
                if SYMBOL_RESOLVER_AVAILABLE:
                    self.symbol_resolver = SymbolResolver(self.current_project)
                    # Simplified analysis loop
                    for f in self.code_analyzer.files[:100]:
                        content = self.code_analyzer.read_file_content(f)
                        if content: self.symbol_resolver.analyze_file(f, content)
                if QUERY_ANALYZER_AVAILABLE:
                    self.query_analyzer = QueryAnalyzer()
                self.console.print("[green]Intelligent analysis complete![/green]")
            else:
                self.console.print("[yellow]Enable YOLO mode first with :yolo[/yellow]")
        elif command == "clear":
            history.clear()
            self.console.print("[green]Chat history cleared.[/green]")
        elif command == "temp":
            if args:
                try:
                    self.temperature = float(args[0])
                    self.console.print(f"[green]Temperature set to {self.temperature}[/green]")
                except ValueError:
                    self.console.print("[red]Invalid temperature value. Use a number between 0.0 and 2.0[/red]")
            else:
                self.console.print(f"[cyan]Current temperature: {self.temperature}[/cyan]")
        elif command == "tokens":
            if args:
                try:
                    self.max_tokens = int(args[0])
                    self.console.print(f"[green]Max tokens set to {self.max_tokens}[/green]")
                except ValueError:
                    self.console.print("[red]Invalid tokens value. Use an integer.[/red]")
            else:
                self.console.print(f"[cyan]Current max tokens: {self.max_tokens or 'unlimited'}[/cyan]")
        elif command == "system":
            if args:
                system_prompt = " ".join(args)
                history.insert(0, {"role": "system", "content": system_prompt})
                self.console.print(f"[green]System prompt set.[/green]")
            else:
                self.console.print("[red]Please provide a system prompt.[/red]")
        elif command == "help":
            self.console.print(HELP_MSG)
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[dim]Type :help for available commands.[/dim]")
        return True

    def handle_image_generation(self, args):
        # ... (implementation for image generation)
        pass