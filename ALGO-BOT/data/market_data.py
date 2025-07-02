#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market data handling module.

This module is responsible for fetching, processing, and managing
market data from MetaTrader 5.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Import MetaTrader5 module
try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 module not found. Please install it using: pip install MetaTrader5")

# Import local modules
from config.mt5_config import MT5Config


class MarketData:
    """Class for handling market data from MT5."""
    
    def __init__(self, mt5_config: MT5Config):
        """Initialize MarketData class.
        
        Args:
            mt5_config: MT5Config instance for connection management
        """
        self.logger = logging.getLogger('main_logger')
        self.mt5_config = mt5_config
        self.symbols = mt5_config.get_symbols()
        self.timeframes = mt5_config.get_timeframes()
        
        # Cache for market data
        self._data_cache = {}
        self._last_update = {}
        
        # Market schedule cache
        self._market_schedule = {}
        
        # Initialize data cache
        self.update()
    
    def update(self) -> None:
        """Update market data for all symbols and timeframes."""
        if not self.mt5_config.is_connected():
            self.logger.warning("Cannot update market data: MT5 not connected")
            return
        
        current_time = datetime.now()
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                # Check if update is needed (avoid excessive API calls)
                cache_key = f"{symbol}_{timeframe}"
                last_update = self._last_update.get(cache_key)
                
                # Update if no data or enough time has passed since last update
                if last_update is None or (current_time - last_update).total_seconds() > self._get_update_interval(timeframe):
                    self._update_symbol_data(symbol, timeframe)
    
    def _get_update_interval(self, timeframe: int) -> int:
        """Get appropriate update interval based on timeframe.
        
        Args:
            timeframe: MT5 timeframe constant
            
        Returns:
            Update interval in seconds
        """
        # Map timeframes to appropriate update intervals
        timeframe_intervals = {
            mt5.TIMEFRAME_M1: 10,      # Update every 10 seconds
            mt5.TIMEFRAME_M5: 30,      # Update every 30 seconds
            mt5.TIMEFRAME_M15: 60,     # Update every minute
            mt5.TIMEFRAME_M30: 120,    # Update every 2 minutes
            mt5.TIMEFRAME_H1: 300,     # Update every 5 minutes
            mt5.TIMEFRAME_H4: 900,     # Update every 15 minutes
            mt5.TIMEFRAME_D1: 3600,    # Update every hour
            mt5.TIMEFRAME_W1: 14400,   # Update every 4 hours
            mt5.TIMEFRAME_MN1: 86400,  # Update every day
        }
        
        return timeframe_intervals.get(timeframe, 60)  # Default to 60 seconds
    
    def _update_symbol_data(self, symbol: str, timeframe: int) -> None:
        """Update market data for a specific symbol and timeframe.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: MT5 timeframe constant
        """
        try:
            # Determine how many bars to fetch based on timeframe
            bars_to_fetch = self._get_bars_count(timeframe)
            
            # Fetch data from MT5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars_to_fetch)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data received for {symbol} on timeframe {timeframe}")
                return
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time in seconds to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            # Add additional columns for analysis
            df['body_size'] = abs(df['close'] - df['open'])
            df['candle_size'] = df['high'] - df['low']
            df['upper_wick'] = df.apply(lambda x: x['high'] - max(x['open'], x['close']), axis=1)
            df['lower_wick'] = df.apply(lambda x: min(x['open'], x['close']) - x['low'], axis=1)
            
            # Calculate candle direction (bullish/bearish)
            df['direction'] = np.where(df['close'] > df['open'], 1, -1)
            df.loc[df['close'] == df['open'], 'direction'] = 0
            
            # Store in cache
            cache_key = f"{symbol}_{timeframe}"
            self._data_cache[cache_key] = df
            self._last_update[cache_key] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol} on timeframe {timeframe}: {e}")
    
    def _get_bars_count(self, timeframe: int) -> int:
        """Determine how many bars to fetch based on timeframe.
        
        Args:
            timeframe: MT5 timeframe constant
            
        Returns:
            Number of bars to fetch
        """
        # Map timeframes to appropriate bar counts
        timeframe_bars = {
            mt5.TIMEFRAME_M1: 1000,    # 1000 minutes
            mt5.TIMEFRAME_M5: 1000,    # 5000 minutes
            mt5.TIMEFRAME_M15: 1000,   # 15000 minutes
            mt5.TIMEFRAME_M30: 1000,   # 30000 minutes
            mt5.TIMEFRAME_H1: 1000,    # 1000 hours
            mt5.TIMEFRAME_H4: 500,     # 2000 hours
            mt5.TIMEFRAME_D1: 365,     # 365 days
            mt5.TIMEFRAME_W1: 100,     # 100 weeks
            mt5.TIMEFRAME_MN1: 36,     # 36 months
        }
        
        return timeframe_bars.get(timeframe, 500)  # Default to 500 bars
    
    def get_data(self, symbol: str, timeframe: int, bars: int = None) -> Optional[pd.DataFrame]:
        """Get market data for a specific symbol and timeframe.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: MT5 timeframe constant
            bars: Number of bars to return (from most recent)
            
        Returns:
            DataFrame with market data or None if not available
        """
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key not in self._data_cache:
            self._update_symbol_data(symbol, timeframe)
        
        if cache_key in self._data_cache:
            df = self._data_cache[cache_key]
            if bars is not None and bars > 0 and bars < len(df):
                return df.iloc[-bars:].copy()
            return df.copy()
        
        return None
        
    def get_historical_data(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> Optional[List]:
        """Get historical market data for a specific symbol and timeframe between dates.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: MT5 timeframe constant
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of rate dictionaries or None if not available
        """
        if not self.mt5_config.is_connected():
            self.logger.warning("Cannot get historical data: MT5 not connected")
            return None
            
        try:
            # Convert datetime to MT5 format
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
            
            # Fetch data from MT5
            rates = mt5.copy_rates_range(symbol, timeframe, 
                                         start_date.to_pydatetime(), 
                                         end_date.to_pydatetime())
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No historical data received for {symbol} on timeframe {timeframe}")
                return None
                
            return rates
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol} on timeframe {timeframe}: {e}")
            return None
            
    def get_latest_bars(self, symbol: str, timeframe: int, count: int = 100) -> Optional[pd.DataFrame]:
        """Get the latest bars for a symbol and timeframe.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: MT5 timeframe constant
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with the latest bars or None if not available
        """
        if not self.mt5_config.is_connected():
            self.logger.warning("Cannot get latest bars: MT5 not connected")
            return None
            
        try:
            # Fetch data from MT5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data received for {symbol} on timeframe {timeframe}")
                return None
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time in seconds to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting latest bars for {symbol} on timeframe {timeframe}: {e}")
            return None
    
    def is_market_open(self, symbol: str = None) -> bool:
        """Check if market is currently open for trading.
        
        Args:
            symbol: Symbol to check (if None, checks first symbol in list)
            
        Returns:
            True if market is open, False otherwise
        """
        if not self.mt5_config.is_connected():
            return False
        
        # Use first symbol if none specified
        if symbol is None:
            if not self.symbols:
                return False
            symbol = self.symbols[0]
        
        # Check if we have cached schedule info that's still valid
        current_time = datetime.now()
        if symbol in self._market_schedule:
            schedule_info, last_check = self._market_schedule[symbol]
            # Use cached info if it's less than 1 hour old
            if (current_time - last_check).total_seconds() < 3600:
                # Check if current time is within trading hours
                if schedule_info:
                    for session in schedule_info:
                        if session['from'] <= current_time.time() <= session['to']:
                            return True
                    return False
        
        # Get fresh schedule information
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Symbol {symbol} not found")
                return False
            
            # Check if symbol is enabled for trading
            if not symbol_info.visible:
                self.logger.warning(f"Symbol {symbol} is not visible")
                return False
            
            # Get trading sessions for current day
            sessions = mt5.symbol_info_tick(symbol)
            if sessions is None:
                self.logger.warning(f"Could not get tick info for {symbol}")
                return False
            
            # If we can get a current tick, the market is open
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current bid/ask price for a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            
        Returns:
            Dictionary with bid and ask prices or None if not available
        """
        if not self.mt5_config.is_connected():
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.warning(f"Could not get tick info for {symbol}")
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_atr(self, symbol: str, timeframe: int, period: int = 14) -> Optional[float]:
        """Calculate Average True Range (ATR) for a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: MT5 timeframe constant
            period: ATR period
            
        Returns:
            ATR value or None if not available
        """
        df = self.get_data(symbol, timeframe)
        if df is None or len(df) < period + 1:
            return None
        
        # Calculate true range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(period).mean()
        
        # Return current ATR value
        return df['atr'].iloc[-1]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            
        Returns:
            Dictionary with symbol information or None if not available
        """
        if not self.mt5_config.is_connected():
            return None
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Symbol {symbol} not found")
                return None
            
            # Convert named tuple to dictionary
            return symbol_info._asdict()
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize MT5 connection
    mt5_config = MT5Config()
    if mt5_config.initialize_mt5():
        # Create market data instance
        market_data = MarketData(mt5_config)
        
        # Get data for EURUSD on H1 timeframe
        df = market_data.get_data("EURUSD", mt5.TIMEFRAME_H1, 10)
        if df is not None:
            print("\nEURUSD H1 Data (last 10 bars):")
            print(df)
        
        # Get current price
        price = market_data.get_current_price("EURUSD")
        if price is not None:
            print(f"\nCurrent EURUSD price: Bid={price['bid']}, Ask={price['ask']}, Spread={price['spread']} points")
        
        # Get ATR
        atr = market_data.get_atr("EURUSD", mt5.TIMEFRAME_H1)
        if atr is not None:
            print(f"\nEURUSD H1 ATR(14): {atr:.5f}")
        
        # Check if market is open
        is_open = market_data.is_market_open("EURUSD")
        print(f"\nEURUSD market is {'open' if is_open else 'closed'}")
        
        # Clean up
        mt5_config.shutdown()
    else:
        print("Failed to connect to MT5")