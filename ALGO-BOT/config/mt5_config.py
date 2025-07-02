#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MetaTrader 5 configuration and connection management.

This module handles the connection to the MT5 platform and provides
access to account information and trading functions.

Updated to support the mt5_credentials.json file for easier configuration.
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple

# Import MetaTrader5 module
try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 module not found. Please install it using: pip install MetaTrader5")


class MT5Config:
    """Class to manage MT5 connection and configuration."""
    
    def __init__(self, config_file: str = None):
        """Initialize MT5 configuration.
        
        Args:
            config_file: Path to the configuration file. If None, default values are used.
        """
        self.logger = logging.getLogger('main_logger')
        self.connected = False
        self.credentials = None
        
        # Default configuration
        self.config = {
            "timeout": 60000,
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],  # Default symbols to trade
            "timeframes": [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1],
            "max_bars": 5000,
            "retry_attempts": 3,
            "retry_delay": 5
        }
        
        # Load credentials from mt5_credentials.json
        self.load_credentials()
        
        # Load additional configuration from file if provided
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_file: Path to the configuration file.
        """
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
            self.logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
    
    def load_credentials(self) -> bool:
        """Load MT5 credentials from configuration file.
        
        Returns:
            bool: True if credentials were loaded successfully, False otherwise.
        """
        try:
            credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mt5_credentials.json')
            if os.path.exists(credentials_path):
                with open(credentials_path, 'r') as f:
                    self.credentials = json.load(f)
                
                # Update config with values from credentials
                if 'symbols' in self.credentials:
                    self.config['symbols'] = self.credentials['symbols']
                
                if 'timeframes' in self.credentials:
                    # Convert string timeframes to MT5 constants
                    tf_map = {
                        "M1": mt5.TIMEFRAME_M1,
                        "M5": mt5.TIMEFRAME_M5,
                        "M15": mt5.TIMEFRAME_M15,
                        "M30": mt5.TIMEFRAME_M30,
                        "H1": mt5.TIMEFRAME_H1,
                        "H4": mt5.TIMEFRAME_H4,
                        "D1": mt5.TIMEFRAME_D1,
                        "W1": mt5.TIMEFRAME_W1,
                        "MN1": mt5.TIMEFRAME_MN1
                    }
                    self.config['timeframes'] = [tf_map.get(tf, mt5.TIMEFRAME_H1) for tf in self.credentials['timeframes']]
                
                if 'max_bars' in self.credentials:
                    self.config['max_bars'] = self.credentials['max_bars']
                
                if 'retry_attempts' in self.credentials:
                    self.config['retry_attempts'] = self.credentials['retry_attempts']
                
                if 'retry_delay' in self.credentials:
                    self.config['retry_delay'] = self.credentials['retry_delay']
                
                self.logger.info(f"MT5 credentials loaded from {credentials_path}")
                return True
            else:
                self.logger.warning(f"MT5 credentials file not found at {credentials_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading MT5 credentials: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        # Check if credentials are loaded
        if self.credentials is None:
            if not self.load_credentials():
                self.logger.error("Failed to load MT5 credentials. Cannot initialize connection.")
                return False
        
        # Ensure MT5 is not already initialized
        if mt5.terminal_info() is not None:
            mt5.shutdown()
        
        # Initialize MT5 connection with retry mechanism
        for attempt in range(self.config.get("retry_attempts", 3)):
            try:
                # Get connection parameters from credentials
                path = self.credentials['account'].get('path', '')
                login = self.credentials['account'].get('login', 0)
                password = self.credentials['account'].get('password', '')
                server = self.credentials['account'].get('server', '')
                
                # Check if required credentials are provided
                if login == 0 or not password or not server:
                    self.logger.error("MT5 credentials not properly configured. Please check mt5_credentials.json")
                    return False
                
                # Initialize MT5 connection
                init_result = mt5.initialize(
                    path=path if path else None,
                    login=login,
                    password=password,
                    server=server,
                    timeout=self.config.get("timeout", 60000)
                )
                
                if not init_result:
                    error = mt5.last_error()
                    self.logger.error(f"MT5 initialization failed. Error code: {error[0]}, Description: {error[1]}")
                    if attempt < self.config.get("retry_attempts", 3) - 1:
                        self.logger.info(f"Retrying in {self.config.get('retry_delay', 5)} seconds... (Attempt {attempt+1}/{self.config.get('retry_attempts', 3)})")
                        time.sleep(self.config.get("retry_delay", 5))
                        continue
                    return False
                
                # Check connection
                if not mt5.terminal_info():
                    self.logger.error("MT5 terminal info not available. Connection failed.")
                    if attempt < self.config.get("retry_attempts", 3) - 1:
                        self.logger.info(f"Retrying in {self.config.get('retry_delay', 5)} seconds... (Attempt {attempt+1}/{self.config.get('retry_attempts', 3)})")
                        time.sleep(self.config.get("retry_delay", 5))
                        continue
                    return False
                
                # Log account info
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Failed to get account info")
                    if attempt < self.config.get("retry_attempts", 3) - 1:
                        self.logger.info(f"Retrying in {self.config.get('retry_delay', 5)} seconds... (Attempt {attempt+1}/{self.config.get('retry_attempts', 3)})")
                        time.sleep(self.config.get("retry_delay", 5))
                        continue
                    return False
                
                # Connection successful
                self.logger.info(f"Connected to MT5 - Server: {account_info.server}")
                self.logger.info(f"Account: {account_info.login}, Name: {account_info.name}, Balance: {account_info.balance}, Equity: {account_info.equity}")
                self.connected = True
                return True
                
            except Exception as e:
                self.logger.error(f"Error during MT5 initialization: {e}")
                if attempt < self.config.get("retry_attempts", 3) - 1:
                    self.logger.info(f"Retrying in {self.config.get('retry_delay', 5)} seconds... (Attempt {attempt+1}/{self.config.get('retry_attempts', 3)})")
                    time.sleep(self.config.get("retry_delay", 5))
                else:
                    return False
        
        return False
        
    def initialize_mt5(self) -> bool:
        """Legacy method for backward compatibility.
        
        Returns:
            bool: Result of initialize() method.
        """
        self.logger.warning("initialize_mt5() is deprecated, use initialize() instead")
        return self.initialize()
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 connection closed")
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information.
        
        Returns:
            Dict containing account information or None if not connected.
        """
        if not self.connected:
            self.logger.warning("Not connected to MT5. Cannot get account info.")
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return None
        
        # Convert named tuple to dictionary
        return account_info._asdict()
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols to trade.
        
        Returns:
            List of symbol names.
        """
        return self.config["symbols"]
    
    def get_timeframes(self) -> List[int]:
        """Get list of timeframes to analyze.
        
        Returns:
            List of MT5 timeframe constants.
        """
        return self.config["timeframes"]
    
    def get_timeframe_by_name(self, timeframe_name: str) -> int:
        """Convert timeframe name to MT5 timeframe constant.
        
        Args:
            timeframe_name: String representation of timeframe (e.g., "M15", "H1")
            
        Returns:
            MT5 timeframe constant
        """
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return tf_map.get(timeframe_name, mt5.TIMEFRAME_H1)  # Default to H1 if not found
    
    def is_connected(self) -> bool:
        """Check if connected to MT5.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connected and mt5.terminal_info() is not None


# Example usage
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create config and test connection
    config = MT5Config()
    if config.initialize():
        print("Connected to MT5 successfully")
        print(f"Account info: {config.get_account_info()}")
        
        # Display available symbols
        symbols = mt5.symbols_get()
        if symbols is not None:
            print(f"Total symbols available: {len(symbols)}")
            print(f"First 5 symbols: {', '.join([s.name for s in symbols[:5]])}")
        
        # Test data retrieval
        if config.get_symbols():
            symbol = config.get_symbols()[0]
            timeframe = config.get_timeframes()[0]
            print(f"\nTesting data retrieval for {symbol} on timeframe {timeframe}")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10)
            if rates is not None:
                import pandas as pd
                from datetime import datetime
                # Convert to pandas DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                print(f"Data retrieved successfully. Latest bar: {df['time'].iloc[-1]}")
                print(df.tail(1))
            else:
                print(f"Failed to retrieve data for {symbol}")
        
        config.shutdown()
    else:
        print("Failed to connect to MT5")