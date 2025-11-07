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

WELCOME_MSG = """
[bold magenta]╔═══════════════════════════════════════════════════════════╗[/bold magenta]
[bold magenta]║[/bold magenta]  [bold white on magenta] MapleCLI [/bold white on magenta]  [bold cyan]AI-Powered Chat Interface[/bold cyan]         [bold magenta]║[/bold magenta]
[bold magenta]╚═══════════════════════════════════════════════════════════╝[/bold magenta]

[bold white]Chat[/bold white]      [dim]>[/dim] Type your message and press Enter
[bold white]Help[/bold white]      [dim]>[/dim] Type [bold cyan]:help[/bold cyan] for commands
[bold white]Exit[/bold white]      [dim]>[/dim] Type [bold cyan]:exit[/bold cyan] or press Ctrl+C

[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
[yellow]Tip:[/yellow] If model responds in another language, ask it to respond in English
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
"""

HELP_MSG = """
[bold magenta]╔═══════════════════════════════════════════════════════════╗[/bold magenta]
[bold magenta]║[/bold magenta]  [bold white on magenta] Commands [/bold white on magenta]                                            [bold magenta]║[/bold magenta]
[bold magenta]╚═══════════════════════════════════════════════════════════╝[/bold magenta]

[bold cyan]Basic Commands[/bold cyan]
  [bold green]:exit[/bold green] [dim]or[/dim] [bold green]:q[/bold green]       [dim]|[/dim] Exit the chat session
  [bold green]:clear[/bold green]             [dim]|[/dim] Clear chat history
  [bold green]:help[/bold green]              [dim]|[/dim] Show this help message

[bold cyan]Configuration[/bold cyan]
  [bold green]:temp[/bold green] [yellow]<0.0-2.0>[/yellow]   [dim]|[/dim] Set temperature (creativity)
  [bold green]:tokens[/bold green] [yellow]<number>[/yellow]  [dim]|[/dim] Set max tokens (response length)
  [bold green]:system[/bold green] [yellow]<prompt>[/yellow]  [dim]|[/dim] Set system prompt

[bold cyan]YOLO Mode (Code Analysis)[/bold cyan]
  [bold green]:yolo[/bold green]              [dim]|[/dim] Toggle intelligent context mode
  [bold green]:analyze[/bold green]           [dim]|[/dim] Analyze current project
  [bold green]:cd[/bold green] [yellow]<path>[/yellow]         [dim]|[/dim] Switch to project directory

[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
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
                self.console.print("\n[bold magenta]╔═══════════════════════════════════════════════════════════╗[/bold magenta]")
                self.console.print("[bold magenta]║[/bold magenta]  [bold white on magenta] Select Model [/bold white on magenta]                                         [bold magenta]║[/bold magenta]")
                self.console.print("[bold magenta]╚═══════════════════════════════════════════════════════════╝[/bold magenta]\n")
                
                # Display models in a clean list
                for i, m in enumerate(models, 1):
                    model_id = m['id']
                    self.console.print(f"  [dim]{i:3d}.[/dim] [cyan]{model_id}[/cyan]")
                
                # Try to find a good default
                default_model = next((m['id'] for m in models if 'gpt-4o-mini' in m['id'].lower()), 
                                   next((m['id'] for m in models if 'gpt-4o' in m['id'].lower()), models[0]['id']))
                
                self.console.print(f"\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
                self.console.print(f"[bold cyan]>[/bold cyan] Select model [dim](number/name, or Enter for[/dim] [yellow]{default_model}[/yellow][dim])[/dim]")
                try:
                    choice = input("  > ").strip()
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
        
        self.console.print(f"\n[bold green]+[/bold green] [dim]Starting chat with[/dim] [bold cyan]{model}[/bold cyan]")

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
        
        self.console.print("\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
        self.console.print("[bold magenta]Thanks for using MapleCLI![/bold magenta]")
        self.console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")

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
            self.console.print("\n[bold white]┌─[/bold white] [bold cyan]You[/bold cyan]")
            self.console.print("[bold white]└─>[/bold white] ", end="")
            user_input = input().strip()
            
            # Check for multiline input (ending with \)
            while user_input.endswith('\\'):
                user_input = user_input[:-1] + '\n'
                self.console.print("   [bold white]|[/bold white] ", end="")
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
            status = "[bold green]ENABLED[/bold green]" if self.yolo_mode else "[bold red]DISABLED[/bold red]"
            self.console.print(f"[bold white]YOLO mode:[/bold white] {status}")
            if self.yolo_mode and not self.current_project:
                self.current_project = os.getcwd()
                self.console.print(f"[dim]Project:[/dim] [cyan]{self.current_project}[/cyan]")
        elif command == "analyze":
            if self.yolo_mode:
                self.console.print("[bold yellow]Analyzing project...[/bold yellow]")
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
                self.console.print("[bold green]+[/bold green] [green]Analysis complete![/green]")
            else:
                self.console.print("[bold yellow]![/bold yellow]  [yellow]Enable YOLO mode first with[/yellow] [cyan]:yolo[/cyan]")
        elif command == "clear":
            history.clear()
            self.console.print("[bold green]+[/bold green] [green]Chat history cleared[/green]")
        elif command == "temp":
            if args:
                try:
                    self.temperature = float(args[0])
                    self.console.print(f"[bold green]+[/bold green] [green]Temperature:[/green] [cyan]{self.temperature}[/cyan]")
                except ValueError:
                    self.console.print("[bold red]x[/bold red] [red]Invalid value. Use 0.0-2.0[/red]")
            else:
                self.console.print(f"[bold white]Current temperature:[/bold white] [cyan]{self.temperature}[/cyan]")
        elif command == "tokens":
            if args:
                try:
                    self.max_tokens = int(args[0])
                    self.console.print(f"[bold green]+[/bold green] [green]Max tokens:[/green] [cyan]{self.max_tokens}[/cyan]")
                except ValueError:
                    self.console.print("[bold red]x[/bold red] [red]Invalid value. Use an integer[/red]")
            else:
                self.console.print(f"[bold white]Current max tokens:[/bold white] [cyan]{self.max_tokens or 'unlimited'}[/cyan]")
        elif command == "system":
            if args:
                system_prompt = " ".join(args)
                history.insert(0, {"role": "system", "content": system_prompt})
                self.console.print(f"[bold green]+[/bold green] [green]System prompt set[/green]")
            else:
                self.console.print("[bold red]x[/bold red] [red]Please provide a system prompt[/red]")
        elif command == "help":
            self.console.print(HELP_MSG)
        else:
            self.console.print(f"[bold red]x[/bold red] [red]Unknown command:[/red] [yellow]{command}[/yellow]")
            self.console.print(f"[dim]Type[/dim] [cyan]:help[/cyan] [dim]for available commands[/dim]")
        return True

    def handle_image_generation(self, args):
        # ... (implementation for image generation)
        pass