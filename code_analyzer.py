"""
MapleCLI Code Analyzer
"""
import os
import asyncio
import re
from typing import List, Dict, Optional

from rich.console import Console
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from logger import maple_logger, SecurityError

class CodeAnalyzer:
    """Analyzes source code in a project directory."""
    
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        '.m', '.mm', '.sh', '.bash', '.zsh', '.bat', '.cmd', '.ps1', '.sql', '.html', '.css', '.scss',
        '.sass', '.vue', '.svelte', '.json', '.yaml', '.yml', '.xml', '.md',
        '.toml', '.ini', '.cfg', '.conf', '.dockerfile', '.lock', '.txt'
    }
    
    IGNORE_DIRS = {
        'node_modules', '.git', '.venv', 'venv', 'env', '__pycache__',
        '.pytest_cache', 'dist', 'build', 'target', '.idea', '.vscode',
        'vendor', 'tmp', 'temp', '.cache', 'coverage', '.next', 'out',
        '.angular', '.svelte-kit', '.nuxt', '.output', 'bower_components',
        '.dart_tool', 'buck-out', '.gradle', '.mvn'
    }
    
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
        self.key_files: List[str] = []
        
        self.max_file_size = int(os.environ.get('MAPLECLI_MAX_FILE_SIZE', 10 * 1024 * 1024))
        self.max_total_size = int(os.environ.get('MAPLECLI_MAX_TOTAL_SIZE', 100 * 1024 * 1024))
        self.max_depth = int(os.environ.get('MAPLECLI_MAX_DEPTH', 10))
        self.processed_size = 0
        
    def _should_ignore_file(self, filepath: str) -> bool:
        """Check if file should be ignored."""
        basename = os.path.basename(filepath)
        if basename.startswith('.') and basename not in {'.env.example', '.gitignore', '.dockerignore'}:
            return True
        
        ignore_patterns = ['*.lock', '*.log', '*.tmp', '*.swp', '*.bak', '*.pyc', '*.class']
        for pattern in ignore_patterns:
            if pattern.replace('*', '') in basename:
                return True
        return False
    
    def _detect_project_type(self):
        """Detect project type based on marker files."""
        for project_type, markers in self.PROJECT_MARKERS.items():
            for marker in markers:
                if os.path.exists(os.path.join(self.project_path, marker)):
                    self.project_type = project_type
                    return
        self.project_type = 'unknown'
    
    async def scan_project_async(self) -> Dict[str, any]:
        """Asynchronously scans the project directory."""
        if not os.path.isdir(self.project_path):
            self.console.print(f"[bold red]Error: Path is not a directory: {self.project_path}[/bold red]")
            return {}
        
        self.files, self.file_stats, self.total_lines, self.key_files, self.processed_size = [], {}, 0, [], 0
        skipped_files = []
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=self.console) as progress:
            task = progress.add_task("[cyan]Scanning project...", total=None)
            self._detect_project_type()
            
            for root, dirs, files in os.walk(self.project_path):
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                
                rel_root = os.path.relpath(root, self.project_path)
                depth = len(rel_root.split(os.sep)) if rel_root != '.' else 0
                if depth > self.max_depth:
                    dirs.clear()
                    continue
                
                for file in files:
                    progress.update(task, description=f"[cyan]Scanning... {len(self.files)} files")
                    if self._should_ignore_file(file):
                        continue
                    
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.project_path)
                    
                    try:
                        file_size = os.path.getsize(filepath)
                        if file_size > self.max_file_size or self.processed_size + file_size > self.max_total_size:
                            skipped_files.append(f"{rel_path} (size limit)")
                            continue
                        self.processed_size += file_size
                        
                        if any(keyword in file.upper() for keyword in ['README', 'LICENSE', 'CONTRIBUTING']) or \
                           file in ['package.json', 'setup.py', 'main.py', 'index.js']:
                            self.key_files.append(rel_path)
                        
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.CODE_EXTENSIONS:
                            self.files.append(rel_path)
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = sum(1 for _ in f)
                                self.total_lines += lines
                                self.file_stats[ext] = self.file_stats.get(ext, 0) + lines
                    except Exception as e:
                        maple_logger.log_error(e, "scan_project", rel_path)
                        skipped_files.append(f"{rel_path} (read error)")
        
        return {
            'project_path': self.project_path, 'project_type': self.project_type,
            'total_files': len(self.files), 'total_lines': self.total_lines,
            'file_stats': self.file_stats, 'key_files': self.key_files,
            'files': self.files[:100], 'skipped_files': skipped_files,
            'processed_size_mb': round(self.processed_size / (1024 * 1024), 2)
        }

    def scan_project(self) -> Dict[str, any]:
        """Synchronous wrapper for scan_project_async."""
        return asyncio.run(self.scan_project_async())

    def get_project_tree(self, max_depth: int = 3) -> Tree:
        """Creates a visual tree representation of the project structure."""
        tree = Tree(f"[bold cyan]{os.path.basename(self.project_path)}[/bold cyan]")
        
        def add_to_tree(parent_tree: Tree, path: str, current_depth: int):
            if current_depth >= max_depth:
                return
            try:
                items = sorted(os.listdir(path))
                dirs = [item for item in items if os.path.isdir(os.path.join(path, item)) and item not in self.IGNORE_DIRS][:10]
                files = [item for item in items if os.path.isfile(os.path.join(path, item))][:10]
                
                for d in dirs:
                    branch = parent_tree.add(f"[blue]{d}/[/blue]")
                    add_to_tree(branch, os.path.join(path, d), current_depth + 1)
                for f in files:
                    parent_tree.add(f"[green]{f}[/green]" if os.path.splitext(f)[1] in self.CODE_EXTENSIONS else f"[dim]{f}[/dim]")
            except PermissionError:
                pass
        
        add_to_tree(tree, self.project_path, 0)
        return tree

    def read_file_content(self, relative_path: str, max_lines: int = 100) -> Optional[str]:
        """Reads the content of a specific file with security validation."""
        try:
            filepath = self._safe_join_path(self.project_path, relative_path)
            if not filepath or not os.path.isfile(filepath):
                return f"Error: Not a regular file: {relative_path}"
            
            if os.path.getsize(filepath) > self.max_file_size:
                return f"Error: File too large"
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:max_lines]
                content = ''.join(lines)
                if sum(1 for _ in f) > 0:
                    content += "\n... (more lines)"
                return content
        except (SecurityError, Exception) as e:
            maple_logger.log_error(e, "read_file_content", relative_path)
            return f"Error reading file: {e}"

    def _safe_join_path(self, base_path: str, relative_path: str) -> Optional[str]:
        """Prevent path traversal attacks."""
        full_path = os.path.abspath(os.path.join(base_path, os.path.normpath(relative_path)))
        if not full_path.startswith(base_path):
            raise SecurityError("Path traversal detected")
        return full_path

    def generate_summary(self) -> str:
        """Generates a comprehensive text summary for AI context."""
        summary_parts = [
            "=== WORKSPACE CONTEXT ===",
            f"Project: {os.path.basename(self.project_path)} ({self.project_type or 'unknown'})",
            f"Stats: {len(self.files)} files, {self.total_lines:,} lines",
            "\n=== LANGUAGES & FILE TYPES ==="
        ]
        for ext, lines in sorted(self.file_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (lines / self.total_lines * 100) if self.total_lines > 0 else 0
            summary_parts.append(f"  {ext:10s}: {lines:7,} lines ({percentage:5.1f}%)")
        
        if self.key_files:
            summary_parts.append("\n=== KEY FILES ===")
            for file in self.key_files[:10]:
                summary_parts.append(f"  - {file}")
        
        summary_parts.append("\n=== PROJECT STRUCTURE (First 40 files) ===")
        for file in self.files[:40]:
            summary_parts.append(f"  - {file}")
        if len(self.files) > 40:
            summary_parts.append(f"  ... and {len(self.files) - 40} more files")
        
        return "\n".join(summary_parts)