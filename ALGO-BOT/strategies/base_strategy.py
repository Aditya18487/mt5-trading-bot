#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base strategy class that defines the interface for all trading strategies.

This module provides the base class that all strategy implementations should inherit from.
It defines the required methods that the backtesting engine expects.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any


class BaseStrategy:
    """Base class for all trading strategies."""
    
    def run_on_bar(self, symbol: str, timeframe: int, current_time: datetime, 
                  current_bar: pd.Series, previous_bars: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process a new price bar and generate trading signals.
        
        This method is called by the backtesting engine for each bar in the historical data.
        Strategy implementations should override this method to implement their logic.
        
        Args:
            symbol: The symbol being processed
            timeframe: The timeframe being processed
            current_time: The timestamp of the current bar
            current_bar: The current price bar data
            previous_bars: All previous bars up to the current one
            
        Returns:
            Optional dictionary with signals and other information, or None if no action
        """
        raise NotImplementedError("Subclasses must implement run_on_bar()")