"""
Advanced Logging Configuration for Debugging

This module provides comprehensive logging setup for debugging
your RAG chatbot application with different log levels and outputs.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_advanced_logging(
    log_level: str = "DEBUG",
    log_file: str = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup advanced logging configuration for debugging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (if None, uses default)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rag_chatbot_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | Line:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # 1. Console Handler (colored output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # 2. File Handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # 3. Error File Handler (errors only)
    error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    configure_specific_loggers()
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Advanced logging configured - Level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log: {error_file}")

def configure_specific_loggers():
    """Configure specific loggers for different components"""
    
    # RAG Pipeline logger
    rag_logger = logging.getLogger('chatbot.rag_pipeline')
    rag_logger.setLevel(logging.DEBUG)
    
    # Model logger
    model_logger = logging.getLogger('chatbot.models')
    model_logger.setLevel(logging.DEBUG)
    
    # UI logger
    ui_logger = logging.getLogger('ui.gradio_interface')
    ui_logger.setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

def debug_decorator(func):
    """
    Decorator to automatically log function entry/exit and parameters
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function entry
        logger.debug(f"ENTERING {func.__name__}")
        logger.debug(f"Args: {args}")
        logger.debug(f"Kwargs: {kwargs}")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log successful exit
            logger.debug(f"EXITING {func.__name__} successfully")
            logger.debug(f"Result type: {type(result)}")
            
            return result
            
        except Exception as e:
            # Log exception
            logger.error(f"EXCEPTION in {func.__name__}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            raise
    
    return wrapper

def log_performance(func):
    """
    Decorator to log function execution time
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"⏱️ {func.__name__} executed in {execution_time:.3f} seconds")
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper

# Example usage in your RAG pipeline
if __name__ == "__main__":
    # Setup logging
    setup_advanced_logging(log_level="DEBUG")
    
    # Test logging
    logger = logging.getLogger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test decorators
    @debug_decorator
    @log_performance
    def test_function(x, y=None):
        """Test function for debugging"""
        logger.info(f"Processing x={x}, y={y}")
        if x == "error":
            raise ValueError("Test error")
        return f"Result: {x}"
    
    # Test normal execution
    try:
        result = test_function("hello", y="world")
        logger.info(f"Function result: {result}")
    except Exception as e:
        logger.error(f"Function failed: {e}")
    
    # Test error case
    try:
        test_function("error")
    except Exception as e:
        logger.error(f"Expected error caught: {e}")
