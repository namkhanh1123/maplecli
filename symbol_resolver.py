"""
Symbol Resolution and Code Understanding for MapleCLI
Tracks symbols, imports, and call relationships across the codebase.
"""

import os
import re
import ast
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger("maplecli.symbol_resolver")


class SymbolResolver:
    """
    Track symbols, imports, and call relationships.
    Similar to LSP (Language Server Protocol) capabilities.
    """
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.symbols = {}  # symbol_name -> [locations]
        self.imports = {}  # file -> [imported_symbols]
        self.exports = {}  # file -> [exported_symbols]
        self.call_graph = {}  # function -> [called_functions]
    
    def analyze_file(self, filepath: str, content: str):
        """Analyze a file and extract symbols."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.py':
            self.analyze_python_symbols(filepath, content)
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            self.analyze_javascript_symbols(filepath, content)
        elif ext == '.java':
            self.analyze_java_symbols(filepath, content)
    
    def analyze_python_symbols(self, filepath: str, content: str):
        """Extract all symbols from Python file."""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.symbols.setdefault(node.name, []).append({
                        'file': filepath,
                        'line': node.lineno,
                        'type': 'function',
                        'name': node.name
                    })
                elif isinstance(node, ast.ClassDef):
                    self.symbols.setdefault(node.name, []).append({
                        'file': filepath,
                        'line': node.lineno,
                        'type': 'class',
                        'name': node.name
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports.setdefault(filepath, []).append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.imports.setdefault(filepath, []).append(node.module)
        except Exception as e:
            logger.debug(f"Failed to parse Python file {filepath}: {e}")
    
    def analyze_javascript_symbols(self, filepath: str, content: str):
        """Extract symbols from JavaScript/TypeScript file using AST if available."""
        try:
            import esprima
            tree = esprima.parseModule(content, {'loc': True})
            
            for node in tree.body:
                if node.type == 'FunctionDeclaration' and node.id:
                    self.symbols.setdefault(node.id.name, []).append({
                        'file': filepath, 'line': node.loc.start.line,
                        'type': 'function', 'name': node.id.name
                    })
                elif node.type == 'ClassDeclaration' and node.id:
                    self.symbols.setdefault(node.id.name, []).append({
                        'file': filepath, 'line': node.loc.start.line,
                        'type': 'class', 'name': node.id.name
                    })
                elif node.type == 'ImportDeclaration' and node.source:
                    self.imports.setdefault(filepath, []).append(node.source.value)
                elif node.type == 'ExportNamedDeclaration' and node.declaration:
                    if node.declaration.id:
                        self.exports.setdefault(filepath, []).append(node.declaration.id.name)
        except (ImportError, Exception) as e:
            if isinstance(e, ImportError):
                logger.info("`esprima` not found. Falling back to regex for JS/TS analysis. Install with `pip install esprima` for better accuracy.")
            
            # Fallback to regex-based analysis
            lines = content.split('\n')
            func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)'
            class_pattern = r'class\s+(\w+)'
            import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
            export_pattern = r'export\s+(?:default\s+)?(?:function|class|const)\s+(\w+)'
            
            for i, line in enumerate(lines):
                func_match = re.search(func_pattern, line)
                if func_match:
                    name = func_match.group(1) or func_match.group(2)
                    if name: self.symbols.setdefault(name, []).append({'file': filepath, 'line': i + 1, 'type': 'function', 'name': name})
                class_match = re.search(class_pattern, line)
                if class_match:
                    name = class_match.group(1)
                    self.symbols.setdefault(name, []).append({'file': filepath, 'line': i + 1, 'type': 'class', 'name': name})
                import_match = re.search(import_pattern, line)
                if import_match:
                    self.imports.setdefault(filepath, []).append(import_match.group(1))
                export_match = re.search(export_pattern, line)
                if export_match:
                    self.exports.setdefault(filepath, []).append(export_match.group(1))
    
    def analyze_java_symbols(self, filepath: str, content: str):
        """Extract symbols from Java file using AST if available."""
        try:
            import javalang
            tree = javalang.parse.parse(content)
            
            for path, node in tree:
                if isinstance(node, javalang.tree.ClassDeclaration):
                    self.symbols.setdefault(node.name, []).append({
                        'file': filepath, 'line': node.position.line,
                        'type': 'class', 'name': node.name
                    })
                elif isinstance(node, javalang.tree.MethodDeclaration):
                    self.symbols.setdefault(node.name, []).append({
                        'file': filepath, 'line': node.position.line,
                        'type': 'method', 'name': node.name
                    })
                elif isinstance(node, javalang.tree.Import):
                    self.imports.setdefault(filepath, []).append(node.path)
        except (ImportError, Exception) as e:
            if isinstance(e, ImportError):
                logger.info("`javalang` not found. Falling back to regex for Java analysis. Install with `pip install javalang` for better accuracy.")

            # Fallback to regex-based analysis
            lines = content.split('\n')
            class_pattern = r'(?:public\s+)?class\s+(\w+)'
            method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)'
            import_pattern = r'import\s+([\w.]+);'
            
            for i, line in enumerate(lines):
                class_match = re.search(class_pattern, line)
                if class_match:
                    name = class_match.group(1)
                    self.symbols.setdefault(name, []).append({'file': filepath, 'line': i + 1, 'type': 'class', 'name': name})
                method_match = re.search(method_pattern, line)
                if method_match:
                    name = method_match.group(1)
                    if name not in ['if', 'for', 'while', 'switch']:
                        self.symbols.setdefault(name, []).append({'file': filepath, 'line': i + 1, 'type': 'method', 'name': name})
                import_match = re.search(import_pattern, line)
                if import_match:
                    self.imports.setdefault(filepath, []).append(import_match.group(1))
    
    def find_definition(self, symbol_name: str) -> List[Dict]:
        """Find where a symbol is defined."""
        return self.symbols.get(symbol_name, [])
    
    def find_usages(self, symbol_name: str, content: str) -> List[int]:
        """Find all usages of a symbol in content."""
        lines = content.split('\n')
        usages = []
        
        for i, line in enumerate(lines):
            if re.search(r'\b' + re.escape(symbol_name) + r'\b', line):
                usages.append(i + 1)
        
        return usages
    
    def get_imports_for_file(self, filepath: str) -> List[str]:
        """Get all imports for a file."""
        return self.imports.get(filepath, [])
    
    def get_exports_for_file(self, filepath: str) -> List[str]:
        """Get all exports for a file."""
        return self.exports.get(filepath, [])
    
    def get_all_symbols(self) -> Dict[str, List[Dict]]:
        """Get all symbols in the project."""
        return self.symbols
    
    def search_symbols(self, query: str) -> List[Dict]:
        """Search for symbols matching a query."""
        results = []
        query_lower = query.lower()
        
        for symbol_name, locations in self.symbols.items():
            if query_lower in symbol_name.lower():
                for loc in locations:
                    results.append({
                        'symbol': symbol_name,
                        'file': loc['file'],
                        'line': loc['line'],
                        'type': loc['type']
                    })
        
        return results
    
    def build_call_graph(self, filepath: str, content: str):
        """Build call graph for a file (simplified version)."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.py':
            self._build_python_call_graph(filepath, content)
    
    def _build_python_call_graph(self, filepath: str, content: str):
        """Build call graph for Python file."""
        try:
            tree = ast.parse(content)
            
            current_function = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    current_function = node.name
                    self.call_graph.setdefault(current_function, [])
                
                elif isinstance(node, ast.Call) and current_function:
                    if isinstance(node.func, ast.Name):
                        called_func = node.func.id
                        if called_func not in self.call_graph[current_function]:
                            self.call_graph[current_function].append(called_func)
        except Exception as e:
            logger.debug(f"Failed to build call graph for {filepath}: {e}")
    
    def get_callers(self, function_name: str) -> List[str]:
        """Find all functions that call a given function."""
        callers = []
        for caller, callees in self.call_graph.items():
            if function_name in callees:
                callers.append(caller)
        return callers
    
    def get_callees(self, function_name: str) -> List[str]:
        """Find all functions called by a given function."""
        return self.call_graph.get(function_name, [])
    
    def generate_dependency_summary(self) -> str:
        """Generate a summary of dependencies."""
        summary_parts = []
        
        summary_parts.append("=== DEPENDENCY SUMMARY ===\n")
        
        # Most imported modules
        all_imports = {}
        for filepath, imports in self.imports.items():
            for imp in imports:
                all_imports[imp] = all_imports.get(imp, 0) + 1
        
        if all_imports:
            summary_parts.append("Top Imported Modules:")
            sorted_imports = sorted(all_imports.items(), key=lambda x: x[1], reverse=True)[:10]
            for module, count in sorted_imports:
                summary_parts.append(f"  {module}: {count} files")
        
        # Symbol statistics
        summary_parts.append(f"\nTotal Symbols: {len(self.symbols)}")
        
        symbol_types = {}
        for symbol_name, locations in self.symbols.items():
            for loc in locations:
                symbol_types[loc['type']] = symbol_types.get(loc['type'], 0) + 1
        
        if symbol_types:
            summary_parts.append("\nSymbol Types:")
            for sym_type, count in sorted(symbol_types.items()):
                summary_parts.append(f"  {sym_type}: {count}")
        
        return "\n".join(summary_parts)

