#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger module for the trading bot.

This module provides:
- Configurable logging to both console and file
- Different log levels for different components
- Rotation of log files
- Formatting of log messages
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path


class BotLogger:
    """Custom logger for the trading bot."""
    
    def __init__(self, log_dir: str = None, log_level: int = logging.INFO):
        """Initialize logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level
        """
        # Set default log directory if not provided
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('main_logger')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        
        # Create file handler
        log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Logger initialized")
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance.
        
        Returns:
            Logger instance
        """
        return self.logger
    
    def set_level(self, level: int) -> None:
        """Set logging level.
        
        Args:
            level: Logging level
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        
        self.logger.info(f"Log level set to {logging.getLevelName(level)}")
    
    def add_component_logger(self, component_name: str, level: int = None) -> logging.Logger:
        """Create a logger for a specific component.
        
        Args:
            component_name: Name of the component
            level: Logging level for the component
        
        Returns:
            Component logger
        """
        if level is None:
            level = self.logger.level
        
        component_logger = logging.getLogger(f'main_logger.{component_name}')
        component_logger.setLevel(level)
        
        # Component logger inherits handlers from parent
        component_logger.propagate = True
        
        return component_logger


def setup_logging(log_dir: str = None, log_level: str or int = 'INFO') -> logging.Logger:
    """Setup logging for the trading bot.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level as string or int
    
    Returns:
        Logger instance
    """
    # Convert string log level to logging constant if it's a string
    if isinstance(log_level, str):
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        level = level_map.get(log_level.upper(), logging.INFO)
    else:
        # If log_level is already an int, use it directly
        level = log_level
    
    # Create and return logger
    bot_logger = BotLogger(log_dir, level)
    return bot_logger.get_logger()