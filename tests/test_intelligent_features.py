"""Tests for intelligent context retrieval features"""
import pytest
import os
import sys
import tempfile
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
try:
    from context_engine import ContextEngine, EMBEDDINGS_AVAILABLE
    from symbol_resolver import SymbolResolver
    from query_analyzer import QueryAnalyzer
    INTELLIGENT_FEATURES_AVAILABLE = True
except ImportError:
    INTELLIGENT_FEATURES_AVAILABLE = False
    pytest.skip("Intelligent features not available", allow_module_level=True)


class TestContextEngine:
    """Test semantic search and code chunking"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.console_mock = Mock()
        self.engine = ContextEngine(self.temp_dir, self.console_mock)
    
    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chunk_python_code(self):
        """Test Python code chunking"""
        python_code = """
def hello():
    '''Say hello'''
    return "Hello"

class MyClass:
    '''A test class'''
    def method(self):
        pass
"""
        chunks = self.engine._chunk_python('test.py', python_code)
        
        assert len(chunks) >= 2  # Should find function and class
        assert any(c['type'] == 'function' for c in chunks)
        assert any(c['type'] == 'class' for c in chunks)
        assert any(c['name'] == 'hello' for c in chunks)
        assert any(c['name'] == 'MyClass' for c in chunks)
    
    def test_chunk_javascript_code(self):
        """Test JavaScript code chunking"""
        js_code = """
function greet(name) {
    return `Hello ${name}`;
}

const add = (a, b) => a + b;

class Calculator {
    multiply(a, b) {
        return a * b;
    }
}
"""
        chunks = self.engine._chunk_javascript('test.js', js_code)
        
        assert len(chunks) >= 3  # Function, arrow function, class
        assert any('greet' in c['name'] for c in chunks)
        assert any('add' in c['name'] for c in chunks)
        assert any('Calculator' in c['name'] for c in chunks)
    
    def test_chunk_by_lines_fallback(self):
        """Test fallback line-based chunking"""
        content = "\n".join([f"line {i}" for i in range(100)])
        chunks = self.engine._chunk_by_lines('test.txt', content, chunk_size=20)
        
        assert len(chunks) == 5  # 100 lines / 20 per chunk
        assert chunks[0]['start_line'] == 1
        assert chunks[0]['end_line'] == 20
    
    @pytest.mark.skipif(not EMBEDDINGS_AVAILABLE, reason="Embeddings not available")
    def test_search_functionality(self):
        """Test semantic search"""
        # Create mock code analyzer
        code_analyzer = Mock()
        code_analyzer.files = ['test.py']
        code_analyzer.read_file_content = Mock(return_value="""
def authenticate_user(username, password):
    '''Authenticate a user with credentials'''
    return check_credentials(username, password)
""")
        
        # Build index
        success = self.engine.build_index(code_analyzer)
        
        if success:
            # Search for authentication
            results = self.engine.search("user authentication", top_k=1)
            
            assert len(results) > 0
            assert 'authenticate' in results[0]['content'].lower()


class TestSymbolResolver:
    """Test symbol resolution and code understanding"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.resolver = SymbolResolver(self.temp_dir)
    
    def test_analyze_python_symbols(self):
        """Test Python symbol extraction"""
        python_code = """
import os
from datetime import datetime

def process_data():
    pass

class DataProcessor:
    def run(self):
        pass
"""
        self.resolver.analyze_python_symbols('test.py', python_code)
        
        # Check symbols
        assert 'process_data' in self.resolver.symbols
        assert 'DataProcessor' in self.resolver.symbols
        
        # Check imports
        assert 'test.py' in self.resolver.imports
        assert 'os' in self.resolver.imports['test.py']
        assert 'datetime' in self.resolver.imports['test.py']
    
    def test_analyze_javascript_symbols(self):
        """Test JavaScript symbol extraction"""
        js_code = """
import React from 'react';
import axios from 'axios';

function App() {
    return <div>Hello</div>;
}

const fetchData = async () => {
    return await axios.get('/api/data');
};

export default App;
"""
        self.resolver.analyze_javascript_symbols('App.jsx', js_code)
        
        # Check symbols
        assert 'App' in self.resolver.symbols
        assert 'fetchData' in self.resolver.symbols
        
        # Check imports
        assert 'App.jsx' in self.resolver.imports
        assert any('react' in imp for imp in self.resolver.imports['App.jsx'])
        
        # Check exports
        assert 'App.jsx' in self.resolver.exports
    
    def test_find_definition(self):
        """Test finding symbol definitions"""
        python_code = """
def my_function():
    pass
"""
        self.resolver.analyze_python_symbols('test.py', python_code)
        
        definitions = self.resolver.find_definition('my_function')
        
        assert len(definitions) == 1
        assert definitions[0]['file'] == 'test.py'
        assert definitions[0]['type'] == 'function'
    
    def test_find_usages(self):
        """Test finding symbol usages"""
        code = """
def helper():
    pass

def main():
    helper()
    result = helper()
    return helper
"""
        usages = self.resolver.find_usages('helper', code)
        
        assert len(usages) >= 3  # Should find all 3 usages
    
    def test_search_symbols(self):
        """Test symbol search"""
        self.resolver.analyze_python_symbols('test.py', """
def user_login():
    pass

def user_logout():
    pass

def admin_login():
    pass
""")
        
        results = self.resolver.search_symbols('user')
        
        assert len(results) >= 2  # user_login and user_logout
        assert any('user_login' in r['symbol'] for r in results)
        assert any('user_logout' in r['symbol'] for r in results)


class TestQueryAnalyzer:
    """Test query intent classification"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = QueryAnalyzer()
    
    def test_find_function_intent(self):
        """Test function finding intent"""
        queries = [
            "Where is the login function?",
            "Find the authenticate function",
            "Show me the process_data function"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze(query)
            assert analysis['intent'] == 'find_function'
            assert len(analysis['entities']) > 0
    
    def test_find_class_intent(self):
        """Test class finding intent"""
        queries = [
            "Where is the User class?",
            "Find the DataProcessor class"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze(query)
            assert analysis['intent'] == 'find_class'
    
    def test_find_usages_intent(self):
        """Test usage finding intent"""
        queries = [
            "Where is authenticate used?",
            "Who calls the save method?",
            "Find usages of User"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze(query)
            assert analysis['intent'] == 'find_usages'
    
    def test_explain_code_intent(self):
        """Test code explanation intent"""
        queries = [
            "Explain the authentication system",
            "What does the login function do?",
            "How does payment processing work?"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze(query)
            assert analysis['intent'] == 'explain_code'
    
    def test_architecture_intent(self):
        """Test architecture intent"""
        queries = [
            "What's the project architecture?",
            "Show me the design patterns",
            "How is the project structured?"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze(query)
            assert analysis['intent'] == 'architecture'
    
    def test_find_bugs_intent(self):
        """Test bug finding intent"""
        queries = [
            "Find security issues",
            "Review for vulnerabilities",
            "Find bugs in the code"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze(query)
            assert analysis['intent'] == 'find_bugs'
    
    def test_general_intent(self):
        """Test general queries"""
        query = "Tell me about this project"
        analysis = self.analyzer.analyze(query)
        
        assert analysis['intent'] == 'general'
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        query = "How does the authentication system work in this project?"
        keywords = self.analyzer.get_search_keywords(query)
        
        assert 'authentication' in keywords
        assert 'system' in keywords
        assert 'work' in keywords
        assert 'project' in keywords
        # Stop words should be removed
        assert 'the' not in keywords
        assert 'in' not in keywords
    
    def test_should_use_semantic_search(self):
        """Test semantic search decision"""
        analysis = {'intent': 'explain_code', 'entities': []}
        assert self.analyzer.should_use_semantic_search(analysis) == True
        
        analysis = {'intent': 'find_function', 'entities': ['login']}
        assert self.analyzer.should_use_semantic_search(analysis) == False
    
    def test_should_use_symbol_search(self):
        """Test symbol search decision"""
        analysis = {'intent': 'find_function', 'entities': ['login']}
        assert self.analyzer.should_use_symbol_search(analysis) == True
        
        analysis = {'intent': 'explain_code', 'entities': []}
        assert self.analyzer.should_use_symbol_search(analysis) == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

