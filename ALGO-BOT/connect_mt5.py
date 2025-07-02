#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to connect to MetaTrader 5 using the provided credentials.
This script will connect to MT5, display account information, and verify that
the connection is working properly.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MT5
try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package is not installed!")
    print("Please install it using: pip install MetaTrader5")
    sys.exit(1)

# Import local modules
from config.mt5_config import MT5Config
from utils.logger import setup_logging


def connect_to_mt5():
    """Connect to MT5 and display account information."""
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Connecting to MT5")
    
    # Initialize MT5 connection
    mt5_config = MT5Config()
    if not mt5_config.initialize():
        logger.error("Failed to initialize MT5")
        return False
    
    try:
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            try:
                terminal_info_dict = terminal_info._asdict()
                logger.info(f"Connected to MT5 Terminal:")
                logger.info(f"  Version: {terminal_info_dict.get('version', 'N/A')}")
                logger.info(f"  Path: {terminal_info_dict.get('path', 'N/A')}")
                logger.info(f"  Connected: {terminal_info_dict.get('connected', 'N/A')}")
            except Exception as e:
                logger.warning(f"Could not process terminal info: {e}")
                logger.info("Connected to MT5 Terminal (details not available)")
        else:
            logger.warning("Terminal info not available")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            mt5_config.shutdown()
            return False
        
        # Convert to dictionary
        account_info = account_info._asdict()
        
        logger.info(f"\nAccount Information:")
        logger.info(f"  Login: {account_info['login']}")
        logger.info(f"  Server: {account_info['server']}")
        logger.info(f"  Name: {account_info['name']}")
        logger.info(f"  Currency: {account_info['currency']}")
        logger.info(f"  Leverage: 1:{account_info['leverage']}")
        logger.info(f"  Balance: {account_info['balance']:.2f}")
        logger.info(f"  Equity: {account_info['equity']:.2f}")
        logger.info(f"  Margin: {account_info['margin']:.2f}")
        logger.info(f"  Free Margin: {account_info['margin_free']:.2f}")
        
        # Get available symbols
        symbols = mt5.symbols_get()
        symbol_names = [symbol.name for symbol in symbols]
        
        logger.info(f"\nAvailable Symbols: {len(symbol_names)}")
        # Show first 10 symbols
        logger.info(f"  Sample: {', '.join(symbol_names[:10])}...")
        
        # Test data retrieval for configured symbols
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'mt5_credentials.json')
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        
        logger.info("\nTesting data retrieval for configured symbols:")
        for symbol in credentials['symbols']:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 10)
            if rates is not None and len(rates) > 0:
                logger.info(f"  Successfully retrieved data for {symbol}")
                logger.info(f"  Latest bar time: {datetime.fromtimestamp(rates[-1][0])}")
            else:
                logger.warning(f"  Could not retrieve data for {symbol}")
        
        logger.info("\nMT5 connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during MT5 connection test: {e}")
        return False
    finally:
        # Shutdown MT5
        mt5_config.shutdown()


if __name__ == "__main__":
    print("\n===== MetaTrader 5 Connection Test =====\n")
    success = connect_to_mt5()
    
    if success:
        print("\n✅ Connection to MT5 successful!")
        print("You can now use the trading bot with your MT5 account.")
        sys.exit(0)
    else:
        print("\n❌ Connection to MT5 failed!")
        print("Please check your credentials and MT5 installation.")
        sys.exit(1)