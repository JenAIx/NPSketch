"""
Structured Logging Setup for NPSketch AI Training

Provides centralized logging configuration and utilities.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys


# Global logger registry
_loggers = {}


def setup_logging(
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5,
    console_enabled: bool = True,
    file_enabled: bool = True
):
    """
    Setup logging configuration for the entire application.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
        date_format: Date format for timestamps
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_enabled: Enable console logging
        file_enabled: Enable file logging
    
    Example:
        >>> setup_logging(
        ...     log_level="INFO",
        ...     log_file="/app/data/logs/training.log"
        ... )
    """
    # Default formats
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_enabled and log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup message
    root_logger.info(f"Logging initialized: level={log_level}, file={'enabled' if file_enabled else 'disabled'}")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    _loggers[name] = logger
    return logger


def setup_from_config(config):
    """
    Setup logging from configuration object or dictionary.
    
    Args:
        config: ConfigLoader instance or full config dict (with 'logging' key)
    
    Example:
        >>> from config import get_config
        >>> config = get_config()
        >>> setup_from_config(config)  # Pass full config object
    """
    # Handle both ConfigLoader and dict
    if hasattr(config, 'get_section'):
        # ConfigLoader instance
        logging_config = config.get_section('logging')
    elif isinstance(config, dict):
        # Dict with 'logging' key
        logging_config = config.get('logging', {})
    else:
        raise TypeError(f"config must be ConfigLoader or dict, got {type(config)}")
    
    # Extract settings
    log_level = logging_config.get('level', 'INFO')
    log_format = logging_config.get('format')
    date_format = logging_config.get('date_format')
    
    # File settings
    file_config = logging_config.get('file', {})
    file_enabled = file_config.get('enabled', True)
    log_file = file_config.get('path')
    max_bytes = file_config.get('max_bytes', 10485760)
    backup_count = file_config.get('backup_count', 5)
    
    # Console settings
    console_config = logging_config.get('console', {})
    console_enabled = console_config.get('enabled', True)
    
    # Setup logging
    setup_logging(
        log_level=log_level,
        log_format=log_format,
        date_format=date_format,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_enabled=console_enabled,
        file_enabled=file_enabled
    )
    
    # Set module-specific levels
    module_loggers = logging_config.get('loggers', {})
    for module_name, module_level in module_loggers.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, module_level.upper()))


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter with context.
    
    Adds contextual information to log messages.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> adapted = LoggerAdapter(logger, {'model_id': 'resnet18'})
        >>> adapted.info("Training started")
        # Output: ... - INFO - [model_id=resnet18] Training started
    """
    
    def process(self, msg, kwargs):
        """Process log message with context."""
        if self.extra:
            context_str = ', '.join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"[{context_str}] {msg}"
        return msg, kwargs


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def train_model(epochs):
        ...     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log execution time.
    
    Args:
        logger: Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> @log_execution_time(logger)
        ... def expensive_operation():
        ...     pass
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()
            logger.debug(f"Starting {func_name}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


# Convenience functions for quick logging
def debug(msg: str, logger_name: str = "npsketch"):
    """Quick debug log."""
    get_logger(logger_name).debug(msg)


def info(msg: str, logger_name: str = "npsketch"):
    """Quick info log."""
    get_logger(logger_name).info(msg)


def warning(msg: str, logger_name: str = "npsketch"):
    """Quick warning log."""
    get_logger(logger_name).warning(msg)


def error(msg: str, logger_name: str = "npsketch", exc_info: bool = False):
    """Quick error log."""
    get_logger(logger_name).error(msg, exc_info=exc_info)


def critical(msg: str, logger_name: str = "npsketch", exc_info: bool = False):
    """Quick critical log."""
    get_logger(logger_name).critical(msg, exc_info=exc_info)

