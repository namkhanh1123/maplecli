"""Security tests for MapleCLI"""
import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the classes we need to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ConfigManager, CodeAnalyzer, SecurityError, MapleLogger


class TestConfigManagerSecurity:
    """Test security features of ConfigManager"""
    
    def test_secure_config_directory_creation(self):
        """Test that config directory is created in secure location"""
        console_mock = Mock()
        
        with patch('platform.system', return_value='Linux'):
            config_manager = ConfigManager(console_mock)
            expected_path = os.path.join(os.path.expanduser('~'), '.config', 'maplecli')
            assert config_manager.config_dir == expected_path
        
        with patch('platform.system', return_value='Windows'):
            config_manager = ConfigManager(console_mock)
            expected_path = os.path.join(os.environ.get('APPDATA', ''), 'maplecli')
            assert config_manager.config_dir == expected_path
        
        with patch('platform.system', return_value='Darwin'):
            config_manager = ConfigManager(console_mock)
            expected_path = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'maplecli')
            assert config_manager.config_dir == expected_path
    
    def test_secure_api_key_input(self):
        """Test that API key input is hidden"""
        console_mock = Mock()
        config_manager = ConfigManager(console_mock)
        
        with patch('getpass.getpass', return_value='hidden_key'):
            with patch('builtins.input', return_value='test_url'):
                config_manager.api_base = None
                config_manager.api_key = None
                config_manager.load_config()
                assert config_manager.api_key == 'hidden_key'
    
    def test_fallback_to_visible_input(self):
        """Test fallback to visible input when getpass fails"""
        console_mock = Mock()
        config_manager = ConfigManager(console_mock)
        
        with patch('getpass.getpass', side_effect=Exception("getpass failed")):
            with patch('builtins.input', return_value='visible_key'):
                with patch.object(console_mock, 'print'):
                    config_manager.api_base = None
                    config_manager.api_key = None
                    config_manager.load_config()
                    assert config_manager.api_key == 'visible_key'
    
    def test_atomic_config_save(self):
        """Test that config is saved atomically"""
        console_mock = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(console_mock)
            config_manager.config_dir = temp_dir
            config_manager.config_file = os.path.join(temp_dir, 'config.json')
            config_manager.api_base = 'test_url'
            config_manager.api_key = 'test_key'
            
            config_manager.save_config()
            
            # Verify file was created
            assert os.path.exists(config_manager.config_file)
            
            # Verify content
            with open(config_manager.config_file, 'r') as f:
                data = json.load(f)
                assert data['OPENAI_API_BASE'] == 'test_url'
                assert data['OPENAI_API_KEY'] == 'test_key'


class TestCodeAnalyzerSecurity:
    """Test security features of CodeAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.console_mock = Mock()
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CodeAnalyzer(self.temp_dir, self.console_mock)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_path_traversal_prevention(self):
        """Test that path traversal attacks are prevented"""
        # Test direct path traversal
        with pytest.raises(SecurityError):
            self.analyzer._safe_join_path(self.temp_dir, '../../../etc/passwd')
        
        # Test encoded path traversal
        with pytest.raises(SecurityError):
            self.analyzer._safe_join_path(self.temp_dir, '..%2F..%2F..%2Fetc%2Fpasswd')
        
        # Test absolute path
        with pytest.raises(SecurityError):
            self.analyzer._safe_join_path(self.temp_dir, '/etc/passwd')
    
    def test_safe_path_validation(self):
        """Test safe path validation"""
        # Test valid relative path
        safe_path = self.analyzer._safe_join_path(self.temp_dir, 'src/main.py')
        expected = os.path.join(self.temp_dir, 'src', 'main.py')
        assert safe_path == expected
        
        # Test valid nested path
        safe_path = self.analyzer._safe_join_path(self.temp_dir, 'src/utils/helper.py')
        expected = os.path.join(self.temp_dir, 'src', 'utils', 'helper.py')
        assert safe_path == expected
    
    def test_file_size_limits(self):
        """Test file size limits are enforced"""
        # Create a large file
        large_file = os.path.join(self.temp_dir, 'large.py')
        with open(large_file, 'w') as f:
            f.write('x' * (11 * 1024 * 1024))  # 11MB file
        
        result = self.analyzer.read_file_content('large.py')
        assert 'too large' in result
    
    def test_regular_file_validation(self):
        """Test that only regular files are processed"""
        # Create a directory
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        os.makedirs(test_dir)
        
        result = self.analyzer.read_file_content('test_dir')
        assert 'Not a regular file' in result
    
    def test_max_total_size_enforcement(self):
        """Test that total size limits are enforced"""
        # Create multiple files that exceed the total limit
        for i in range(12):  # 12 files of 10MB each = 120MB > 100MB limit
            large_file = os.path.join(self.temp_dir, f'large_{i}.py')
            with open(large_file, 'w') as f:
                f.write('x' * (10 * 1024 * 1024))  # 10MB each
        
        # This should trigger the size limit during scanning
        result = asyncio.run(self.analyzer.scan_project_async())
        assert 'skipped_files' in result
        assert len(result['skipped_files']) > 0


class TestSecurityLogging:
    """Test security logging functionality"""
    
    def test_security_event_logging(self):
        """Test that security events are properly logged"""
        logger = MapleLogger()
        
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_security_event("Test Event", "Test details")
            mock_warning.assert_called_once_with("SECURITY: Test Event - Test details")
    
    def test_error_logging_with_context(self):
        """Test error logging with structured context"""
        logger = MapleLogger()
        test_error = ValueError("Test error")
        
        with patch.object(logger.logger, 'error') as mock_error:
            logger.log_error(test_error, "test_operation", "test_file.py", "high")
            
            # Verify the error was logged with correct context
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            assert "test_operation" in call_args
            assert "Test error" in call_args


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        console_mock = Mock()
        config_manager = ConfigManager(console_mock)
        
        # Test invalid API base URL
        with patch('builtins.input', return_value='not-a-url'):
            with patch.object(console_mock, 'print'):
                config_manager.api_base = None
                config_manager.load_config()
                # Should still accept but warn about invalid format
    
    def test_file_extension_filtering(self):
        """Test that only allowed file extensions are processed"""
        console_mock = Mock()
        temp_dir = tempfile.mkdtemp()
        
        try:
            analyzer = CodeAnalyzer(temp_dir, console_mock)
            
            # Create test files with different extensions
            allowed_file = os.path.join(temp_dir, 'test.py')
            disallowed_file = os.path.join(temp_dir, 'test.exe')
            
            with open(allowed_file, 'w') as f:
                f.write('print("hello")')
            with open(disallowed_file, 'w') as f:
                f.write('binary content')
            
            # Test that only allowed extensions are processed
            assert '.py' in analyzer.CODE_EXTENSIONS
            assert '.exe' not in analyzer.CODE_EXTENSIONS
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    pytest.main([__file__])