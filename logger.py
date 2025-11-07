"""
MapleCLI Logger and Custom Exceptions
"""
import logging
from typing import Optional

# Security and logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maplecli.log'),
        logging.StreamHandler()
    ]
)

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

class MapleLogger:
    """Enhanced logging with structured error context"""
    def __init__(self, name: str = "maplecli"):
        self.logger = logging.getLogger(name)
        
    def log_error(self, error: Exception, operation: str, filepath: Optional[str] = None, severity: str = "medium"):
        """Structured error logging"""
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            extra={
                'severity': severity,
                'filepath': filepath,
                'error_type': type(error).__name__
            }
        )
        
    def log_security_event(self, event: str, details: str):
        """Log security-related events"""
        self.logger.warning(f"SECURITY: {event} - {details}")

# Global logger instance
maple_logger = MapleLogger()