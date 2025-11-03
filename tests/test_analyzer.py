"""Code analysis tests for MapleCLI"""
import pytest
import os
import tempfile
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

# Import the classes we need to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import CodeAnalyzer


class TestCodeAnalyzer:
    """Test code analysis functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.console_mock = Mock()
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CodeAnalyzer(self.temp_dir, self.console_mock)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str):
        """Helper to create test files"""
        filepath = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def test_project_type_detection(self):
        """Test project type detection"""
        # Python project
        self.create_test_file('requirements.txt', 'requests==2.25.0')
        self.create_test_file('setup.py', 'from setuptools import setup')
        self.analyzer._detect_project_type()
        assert self.analyzer.project_type == 'python'
        
        # Node.js project
        self.create_test_file('package.json', '{"name": "test"}')
        self.analyzer._detect_project_type()
        assert self.analyzer.project_type == 'node'
        
        # Rust project
        self.create_test_file('Cargo.toml', '[package]\nname = "test"')
        self.analyzer._detect_project_type()
        assert self.analyzer.project_type == 'rust'
    
    def test_file_filtering(self):
        """Test that files are properly filtered"""
        # Should be ignored
        assert self.analyzer._should_ignore_file('.git/config')
        assert self.analyzer._should_ignore_file('node_modules/package.json')
        assert self.analyzer._should_ignore_file('test.pyc')
        assert self.analyzer._should_ignore_file('test.log')
        
        # Should not be ignored
        assert not self.analyzer._should_ignore_file('src/main.py')
        assert not self.analyzer._should_ignore_file('.env.example')
        assert not self.analyzer._should_ignore_file('.gitignore')
    
    @pytest.mark.asyncio
    async def test_async_project_scan(self):
        """Test async project scanning"""
        # Create test files
        self.create_test_file('src/main.py', 'print("hello")')
        self.create_test_file('src/utils.py', 'def helper(): pass')
        self.create_test_file('README.md', '# Test Project')
        self.create_test_file('package.json', '{"name": "test"}')
        
        result = await self.analyzer.scan_project_async()
        
        assert result['total_files'] >= 2  # At least the Python files
        assert result['project_type'] == 'node'  # Should detect Node.js
        assert 'processed_size_mb' in result
        assert 'skipped_files' in result
    
    def test_python_dependency_analysis(self):
        """Test Python dependency extraction"""
        python_code = """
import os
import sys
from datetime import datetime
from collections import defaultdict
import requests
import numpy as np
"""
        self.create_test_file('test.py', python_code)
        
        deps = self.analyzer.analyze_dependencies('test.py')
        
        expected_imports = ['os', 'sys', 'datetime', 'collections', 'requests', 'numpy']
        for imp in expected_imports:
            assert imp in deps['imports']
    
    def test_javascript_dependency_analysis(self):
        """Test JavaScript dependency extraction"""
        js_code = """
import React from 'react';
import { useState } from 'react';
import axios from 'axios';
const express = require('express');
import lodash from 'lodash';
"""
        self.create_test_file('test.js', js_code)
        
        deps = self.analyzer.analyze_dependencies('test.js')
        
        expected_imports = ['react', 'axios', 'express', 'lodash']
        for imp in expected_imports:
            assert imp in deps['imports']
    
    def test_java_dependency_analysis(self):
        """Test Java dependency extraction"""
        java_code = """
import java.util.List;
import java.util.ArrayList;
import com.example.MyClass;
import org.springframework.boot.SpringApplication;
"""
        self.create_test_file('Test.java', java_code)
        
        deps = self.analyzer.analyze_dependencies('Test.java')
        
        expected_imports = ['java', 'com', 'org']
        for imp in expected_imports:
            assert imp in deps['imports']
    
    def test_architecture_pattern_detection(self):
        """Test architecture pattern detection"""
        files = [
            'src/controllers/UserController.java',
            'src/models/User.java',
            'src/views/UserView.jsp',
            'services/auth-service.js',
            'api/gateway.js',
            'functions/lambda-handler.js',
            'modules/user/module.py',
            'components/Button.jsx',
            'repositories/UserRepository.java',
            'dao/UserDAO.java'
        ]
        
        patterns = self.analyzer.detect_architecture_patterns(files)
        
        assert patterns['mvc'] >= 3  # controller, model, view
        assert patterns['microservices'] >= 2  # service, gateway
        assert patterns['serverless'] >= 1  # lambda
        assert patterns['modular'] >= 2  # modules, components
        assert patterns['repository'] >= 2  # repository, dao
    
    def test_python_complexity_analysis(self):
        """Test Python complexity analysis"""
        python_code = """
def simple_function():
    return "hello"

def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                try:
                    result = i * 2
                except Exception:
                    result = 0
            else:
                result = i / 2
    elif x < 0:
        while x < 0:
            x += 1
    return result
"""
        self.create_test_file('test.py', python_code)
        
        complexity = self.analyzer.analyze_complexity('test.py')
        
        assert complexity['functions'] == 2
        assert complexity['lines_of_code'] > 0
        assert complexity['cyclomatic'] > 1  # Should have decision points
        assert complexity['cognitive'] >= 1  # Should have nesting
    
    def test_javascript_complexity_analysis(self):
        """Test JavaScript complexity analysis"""
        js_code = """
function simpleFunction() {
    return "hello";
}

const complexFunction = (x) => {
    if (x > 0) {
        for (let i = 0; i < x; i++) {
            if (i % 2 === 0) {
                try {
                    const result = i * 2;
                } catch (error) {
                    const result = 0;
                }
            } else {
                const result = i / 2;
            }
        }
    } else if (x < 0) {
        while (x < 0) {
            x++;
        }
    }
    return result;
};
"""
        self.create_test_file('test.js', js_code)
        
        complexity = self.analyzer.analyze_complexity('test.js')
        
        assert complexity['functions'] == 2
        assert complexity['lines_of_code'] > 0
        assert complexity['cyclomatic'] > 1
    
    def test_java_complexity_analysis(self):
        """Test Java complexity analysis"""
        java_code = """
public class TestClass {
    public void simpleMethod() {
        System.out.println("hello");
    }
    
    public void complexMethod(int x) {
        if (x > 0) {
            for (int i = 0; i < x; i++) {
                if (i % 2 == 0) {
                    try {
                        int result = i * 2;
                    } catch (Exception e) {
                        int result = 0;
                    }
                } else {
                    int result = i / 2;
                }
            }
        } else if (x < 0) {
            while (x < 0) {
                x++;
            }
        }
    }
}
"""
        self.create_test_file('TestClass.java', java_code)
        
        complexity = self.analyzer.analyze_complexity('TestClass.java')
        
        assert complexity['functions'] >= 2
        assert complexity['lines_of_code'] > 0
        assert complexity['cyclomatic'] > 1
    
    def test_go_dependency_analysis(self):
        """Test Go dependency extraction"""
        go_code = """
package main

import (
    "fmt"
    "net/http"
    "github.com/gin-gonic/gin"
    "database/sql"
)
"""
        self.create_test_file('main.go', go_code)
        
        deps = self.analyzer.analyze_dependencies('main.go')
        
        expected_imports = ['fmt', 'net/http', 'github.com/gin-gonic/gin', 'database/sql']
        for imp in expected_imports:
            assert imp in deps['imports']
    
    def test_rust_dependency_analysis(self):
        """Test Rust dependency extraction"""
        rust_code = """
use std::collections::HashMap;
use std::io::Result;
use serde::{Deserialize, Serialize};
extern crate tokio;
use crate::utils::helper;
"""
        self.create_test_file('main.rs', rust_code)
        
        deps = self.analyzer.analyze_dependencies('main.rs')
        
        expected_imports = ['std', 'serde', 'tokio', 'crate']
        for imp in expected_imports:
            assert imp in deps['imports']
    
    def test_project_tree_generation(self):
        """Test project tree visualization"""
        # Create a test directory structure
        self.create_test_file('src/main.py', 'print("hello")')
        self.create_test_file('src/utils/helper.py', 'def help(): pass')
        self.create_test_file('tests/test_main.py', 'def test(): pass')
        self.create_test_file('README.md', '# Test')
        
        tree = self.analyzer.get_project_tree(max_depth=2)
        
        # The tree should contain the project name
        assert os.path.basename(self.temp_dir) in str(tree)
    
    def test_summary_generation(self):
        """Test project summary generation"""
        # Create test files
        self.create_test_file('src/main.py', 'print("hello")')
        self.create_test_file('src/utils.py', 'def help(): pass')
        self.create_test_file('README.md', '# Test Project')
        
        # Run scan first to populate data
        asyncio.run(self.analyzer.scan_project_async())
        
        summary = self.analyzer.generate_summary()
        
        assert "WORKSPACE CONTEXT" in summary
        assert "PROJECT STATISTICS" in summary
        assert str(self.analyzer.total_files) in summary
        assert str(self.analyzer.total_lines) in summary
    
    def test_error_handling_in_analysis(self):
        """Test error handling during analysis"""
        # Create a file with invalid encoding
        invalid_file = os.path.join(self.temp_dir, 'invalid.py')
        with open(invalid_file, 'wb') as f:
            f.write(b'\xff\xfe invalid content')
        
        # Should handle gracefully
        result = asyncio.run(self.analyzer.scan_project_async())
        assert 'skipped_files' in result
    
    def test_memory_management(self):
        """Test memory management during large scans"""
        # Create many small files
        for i in range(100):
            self.create_test_file(f'src/file_{i}.py', f'def func_{i}(): return {i}')
        
        result = asyncio.run(self.analyzer.scan_project_async())
        
        # Should process files but respect limits
        assert result['total_files'] == 100
        assert result['processed_size_mb'] < 100  # Should be reasonable


if __name__ == '__main__':
    pytest.main([__file__])