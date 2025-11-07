"""
MapleCLI Configuration Manager
"""
import os
import json
import stat
import platform
import getpass
from typing import List, Optional

from rich.console import Console
from logger import maple_logger

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
            
            config_data = {
                "OPENAI_API_BASE": self.api_base,
                "OPENAI_API_KEY": self.api_key,
                "recent_projects": self.recent_projects[-10:]
            }
            
            temp_file = self.config_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            if os.path.exists(self.config_file):
                os.replace(temp_file, self.config_file)
            else:
                os.rename(temp_file, self.config_file)
            
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
                import ctypes
                try:
                    ctypes.windll.kernel32.SetFileAttributesW(filepath, 0x2)
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