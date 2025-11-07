"""
Intelligent Context Engine for MapleCLI
Provides semantic search and intelligent code retrieval similar to Augment CLI.
"""

import os
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger("maplecli.context_engine")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers or faiss not available. Install with: pip install sentence-transformers faiss-cpu")


class ContextEngine:
    """
    Intelligent code retrieval using semantic search.
    Similar to Augment's context engine.
    """
    
    def __init__(self, project_path: str, console=None):
        self.project_path = project_path
        self.console = console
        self.model = None
        self.index = None
        self.chunks = []  # Store code chunks
        self.metadata = []  # Store file paths, line numbers, etc.
        self.cache_dir = os.path.join(project_path, '.maplecli_cache')
        self.index_cache_file = os.path.join(self.cache_dir, 'index.faiss')
        self.metadata_cache_file = os.path.join(self.cache_dir, 'metadata.json')
        self.state_cache_file = os.path.join(self.cache_dir, 'state.json')
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.model = None
        
    def chunk_code(self, filepath: str, content: str) -> List[Dict]:
        """
        Chunk code into semantic units (functions, classes, blocks).
        """
        chunks = []
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.py':
            chunks = self._chunk_python(filepath, content)
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            chunks = self._chunk_javascript(filepath, content)
        elif ext == '.java':
            chunks = self._chunk_java(filepath, content)
        else:
            # Fallback: chunk by lines (every 50 lines)
            chunks = self._chunk_by_lines(filepath, content, chunk_size=50)
        
        return chunks
    
    def _chunk_python(self, filepath: str, content: str) -> List[Dict]:
        """Extract Python functions and classes as chunks."""
        import ast
        chunks = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                    
                    chunk_content = '\n'.join(lines[start_line:end_line])
                    
                    # Get docstring if available
                    docstring = ast.get_docstring(node) or ""
                    
                    chunks.append({
                        'filepath': filepath,
                        'start_line': start_line + 1,
                        'end_line': end_line,
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'name': node.name,
                        'content': chunk_content,
                        'docstring': docstring,
                        'searchable_text': f"{node.name} {docstring} {chunk_content}"
                    })
        except Exception as e:
            logger.debug(f"Failed to parse Python file {filepath}: {e}")
            # Fallback to line-based chunking
            return self._chunk_by_lines(filepath, content)
        
        if not chunks:
            return self._chunk_by_lines(filepath, content)
        
        return chunks
    
    def _chunk_javascript(self, filepath: str, content: str) -> List[Dict]:
        """Extract JavaScript/TypeScript functions and classes using AST if available."""
        try:
            import esprima
            tree = esprima.parseModule(content, {'loc': True})
            lines = content.split('\n')
            chunks = []

            for node in tree.body:
                if node.type in ['FunctionDeclaration', 'ClassDeclaration'] and node.id:
                    start_line = node.loc.start.line - 1
                    end_line = node.loc.end.line
                    chunk_content = '\n'.join(lines[start_line:end_line])
                    chunks.append({
                        'filepath': filepath, 'start_line': start_line + 1, 'end_line': end_line,
                        'type': 'function' if node.type == 'FunctionDeclaration' else 'class',
                        'name': node.id.name, 'content': chunk_content,
                        'searchable_text': f"{node.id.name} {chunk_content}"
                    })
            if chunks: return chunks
        except (ImportError, Exception):
            pass  # Fallback to regex
        
        # Fallback regex implementation
        import re
        chunks = []
        lines = content.split('\n')
        patterns = [
            (r'function\s+(\w+)\s*\([^)]*\)\s*{', 'function'),
            (r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', 'function'),
            (r'class\s+(\w+)\s*{', 'class'),
        ]
        for i, line in enumerate(lines):
            for pattern, chunk_type in patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) if match.groups() else 'anonymous'
                    chunk_lines = lines[i:min(i + 40, len(lines))]
                    chunks.append({
                        'filepath': filepath, 'start_line': i + 1, 'end_line': min(i + 40, len(lines)),
                        'type': chunk_type, 'name': name, 'content': '\n'.join(chunk_lines),
                        'searchable_text': f"{name} {' '.join(chunk_lines)}"
                    })
                    break
        return chunks if chunks else self._chunk_by_lines(filepath, content)
    
    def _chunk_java(self, filepath: str, content: str) -> List[Dict]:
        """Extract Java methods and classes using AST if available."""
        try:
            import javalang
            tree = javalang.parse.parse(content)
            lines = content.split('\n')
            chunks = []

            for path, node in tree:
                if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.MethodDeclaration)):
                    start_line = node.position.line - 1
                    # Estimate end line as javalang doesn't provide it
                    end_line = start_line + len(str(node).split('\n')) + 5
                    chunk_content = '\n'.join(lines[start_line:min(end_line, len(lines))])
                    chunks.append({
                        'filepath': filepath, 'start_line': start_line + 1, 'end_line': min(end_line, len(lines)),
                        'type': 'class' if isinstance(node, javalang.tree.ClassDeclaration) else 'method',
                        'name': node.name, 'content': chunk_content,
                        'searchable_text': f"{node.name} {chunk_content}"
                    })
            if chunks: return chunks
        except (ImportError, Exception):
            pass # Fallback to regex

        # Fallback regex implementation
        import re
        chunks = []
        lines = content.split('\n')
        patterns = [
            (r'class\s+(\w+)', 'class'),
            (r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*{', 'method'),
        ]
        for i, line in enumerate(lines):
            for pattern, chunk_type in patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    chunk_lines = lines[i:min(i + 40, len(lines))]
                    chunks.append({
                        'filepath': filepath, 'start_line': i + 1, 'end_line': min(i + 40, len(lines)),
                        'type': chunk_type, 'name': name, 'content': '\n'.join(chunk_lines),
                        'searchable_text': f"{name} {' '.join(chunk_lines)}"
                    })
                    break
        return chunks if chunks else self._chunk_by_lines(filepath, content)
    
    def _chunk_by_lines(self, filepath: str, content: str, chunk_size: int = 50) -> List[Dict]:
        """Fallback: chunk by fixed line count."""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i+chunk_size]
            chunks.append({
                'filepath': filepath,
                'start_line': i + 1,
                'end_line': min(i + chunk_size, len(lines)),
                'type': 'block',
                'name': f'block_{i}',
                'content': '\n'.join(chunk_lines),
                'searchable_text': '\n'.join(chunk_lines)
            })
        
        return chunks
    
    def build_index(self, code_analyzer) -> bool:
        """
        Build FAISS index from all code chunks.
        Returns True if successful, False otherwise.
        """
        if not EMBEDDINGS_AVAILABLE or not self.model:
            if self.console:
                self.console.print("[yellow]⚠️  Semantic search not available. Install with: pip install sentence-transformers faiss-cpu[/yellow]")
            return False
        
        # Check if cached index exists and is fresh
        if self._load_cached_index(code_analyzer):
            if self.console:
                self.console.print(f"[green]✓ Loaded fresh cache ({len(self.chunks)} chunks)[/green]")
            return True
        
        all_chunks = []
        all_texts = []
        
        if self.console:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[cyan]Building semantic index...", total=len(code_analyzer.files))
                
                for filepath in code_analyzer.files:
                    try:
                        content = code_analyzer.read_file_content(filepath, max_lines=10000)
                        if content and not content.startswith("Error"):
                            chunks = self.chunk_code(filepath, content)
                            all_chunks.extend(chunks)
                            all_texts.extend([c['searchable_text'] for c in chunks])
                    except Exception as e:
                        logger.debug(f"Error chunking {filepath}: {e}")
                    progress.advance(task)
        else:
            for filepath in code_analyzer.files:
                try:
                    content = code_analyzer.read_file_content(filepath, max_lines=10000)
                    if content and not content.startswith("Error"):
                        chunks = self.chunk_code(filepath, content)
                        all_chunks.extend(chunks)
                        all_texts.extend([c['searchable_text'] for c in chunks])
                except Exception as e:
                    logger.debug(f"Error chunking {filepath}: {e}")
        
        if not all_texts:
            if self.console:
                self.console.print("[yellow]⚠️  No code chunks to index[/yellow]")
            return False
        
        # Generate embeddings
        if self.console:
            self.console.print(f"[cyan]Generating embeddings for {len(all_texts)} code chunks...[/cyan]")
        
        try:
            embeddings = self.model.encode(all_texts, show_progress_bar=False, batch_size=32)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return False
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks = all_chunks
        self.metadata = [{'filepath': c['filepath'], 'start_line': c['start_line']} for c in all_chunks]
        
        # Cache the index
        self._save_index(code_analyzer)
        
        if self.console:
            self.console.print(f"[green]✓ Index built: {len(all_chunks)} chunks indexed[/green]")
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant code chunks using semantic similarity.
        """
        if self.index is None or not EMBEDDINGS_AVAILABLE or not self.model:
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Search
            distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))
            
            # Return results with scores
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks) and idx >= 0:
                    chunk = self.chunks[idx].copy()
                    chunk['relevance_score'] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                    results.append(chunk)
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _get_project_state_hash(self, code_analyzer) -> str:
        """Generates a hash representing the current state of project files."""
        try:
            file_metadata = []
            for f in code_analyzer.files:
                full_path = os.path.join(self.project_path, f)
                if os.path.exists(full_path):
                    mtime = os.path.getmtime(full_path)
                    file_metadata.append(f"{f}:{mtime}")
            
            state_string = "".join(sorted(file_metadata))
            return hashlib.md5(state_string.encode()).hexdigest()
        except Exception as e:
            logger.debug(f"Could not generate project state hash: {e}")
            return ""

    def _save_index(self, code_analyzer):
        """Save index and metadata to disk using a secure format."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, self.index_cache_file)

            # Save metadata as JSON
            import json
            with open(self.metadata_cache_file, 'w') as f:
                json.dump({'chunks': self.chunks, 'metadata': self.metadata}, f)

            # Save project state hash
            state_hash = self._get_project_state_hash(code_analyzer)
            with open(self.state_cache_file, 'w') as f:
                json.dump({'hash': state_hash}, f)

        except Exception as e:
            logger.debug(f"Failed to save index cache: {e}")
    
    def _load_cached_index(self, code_analyzer) -> bool:
        """Load cached index if available and fresh."""
        if not all(os.path.exists(p) for p in [self.index_cache_file, self.metadata_cache_file, self.state_cache_file]):
            return False
        
        try:
            # Check if cache is stale
            import json
            with open(self.state_cache_file, 'r') as f:
                stored_state = json.load(f)
            
            current_hash = self._get_project_state_hash(code_analyzer)
            
            if stored_state.get('hash') != current_hash:
                logger.info("Project has changed, rebuilding index.")
                return False

            # Load index and metadata
            self.index = faiss.read_index(self.index_cache_file)
            with open(self.metadata_cache_file, 'r') as f:
                metadata = json.load(f)
            
            self.chunks = metadata['chunks']
            self.metadata = metadata['metadata']
            
            return True
        except Exception as e:
            logger.debug(f"Failed to load cached index: {e}")
            return False

