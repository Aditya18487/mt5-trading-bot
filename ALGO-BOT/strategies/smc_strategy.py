#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Money Concepts (SMC) trading strategy implementation.

This module implements the core SMC strategy logic, including:
- Liquidity zone detection (swing highs/lows, order blocks, FVGs)
- Candlestick pattern recognition
- Market structure analysis
- Trade setup generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime

# Import MetaTrader5 module
import MetaTrader5 as mt5

# Import local modules
from data.market_data import MarketData
from execution.order_manager import OrderManager
from strategies.base_strategy import BaseStrategy


@dataclass
class LiquidityZone:
    """Represents a liquidity zone in the market."""
    symbol: str
    timeframe: int
    zone_type: str  # 'swing_high', 'swing_low', 'order_block_bullish', 'order_block_bearish', 'fvg_bullish', 'fvg_bearish', 'consolidation'
    price_high: float
    price_low: float
    strength: float  # 0-1 scale indicating zone strength
    created_time: datetime
    expiry_time: Optional[datetime] = None
    swept: bool = False  # Whether the zone has been swept by price
    
    @property
    def mid_price(self) -> float:
        """Get the middle price of the zone."""
        return (self.price_high + self.price_low) / 2
    
    @property
    def zone_size(self) -> float:
        """Get the size of the zone in price units."""
        return self.price_high - self.price_low
    
    def is_valid(self) -> bool:
        """Check if the liquidity zone is still valid."""
        if self.swept:
            return False
        
        if self.expiry_time and datetime.now() > self.expiry_time:
            return False
        
        return True
    
    def is_price_inside(self, price: float) -> bool:
        """Check if a price is inside the zone."""
        return self.price_low <= price <= self.price_high
    
    def is_bullish(self) -> bool:
        """Check if the zone is bullish."""
        return self.zone_type in ['swing_low', 'order_block_bullish', 'fvg_bullish']
    
    def is_bearish(self) -> bool:
        """Check if the zone is bearish."""
        return self.zone_type in ['swing_high', 'order_block_bearish', 'fvg_bearish']


@dataclass
class CandlePattern:
    """Represents a candlestick pattern."""
    symbol: str
    timeframe: int
    pattern_type: str  # 'engulfing_bullish', 'engulfing_bearish', 'pin_bar_bullish', 'pin_bar_bearish', 'inside_bar', 'doji'
    bar_index: int  # Index of the pattern in the dataframe
    strength: float  # 0-1 scale indicating pattern strength
    created_time: datetime
    
    def is_bullish(self) -> bool:
        """Check if the pattern is bullish."""
        return self.pattern_type in ['engulfing_bullish', 'pin_bar_bullish']
    
    def is_bearish(self) -> bool:
        """Check if the pattern is bearish."""
        return self.pattern_type in ['engulfing_bearish', 'pin_bar_bearish']


@dataclass
class MarketStructure:
    """Represents market structure analysis."""
    symbol: str
    timeframe: int
    structure_type: str  # 'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish', 'hh_hl', 'll_lh'
    price_level: float
    created_time: datetime
    strength: float  # 0-1 scale indicating structure strength
    
    def is_bullish(self) -> bool:
        """Check if the structure is bullish."""
        return self.structure_type in ['bos_bullish', 'choch_bullish', 'hh_hl']
    
    def is_bearish(self) -> bool:
        """Check if the structure is bearish."""
        return self.structure_type in ['bos_bearish', 'choch_bearish', 'll_lh']


@dataclass
class ChartPattern:
    """Represents a chart pattern."""
    symbol: str
    timeframe: int
    pattern_type: str  # 'double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders', 'triangle', 'flag'
    price_high: float
    price_low: float
    target_price: float
    created_time: datetime
    strength: float  # 0-1 scale indicating pattern strength
    
    def is_bullish(self) -> bool:
        """Check if the pattern is bullish."""
        return self.pattern_type in ['double_bottom', 'inv_head_shoulders', 'flag_bullish']
    
    def is_bearish(self) -> bool:
        """Check if the pattern is bearish."""
        return self.pattern_type in ['double_top', 'head_shoulders', 'flag_bearish']


@dataclass
class TradeSetup:
    """Represents a complete trade setup."""
    symbol: str
    timeframe: int
    direction: int  # 1 for long, -1 for short
    entry_price: float
    stop_loss: float
    take_profit: float
    created_time: datetime
    expiry_time: Optional[datetime] = None
    liquidity_zones: List[LiquidityZone] = None
    candle_patterns: List[CandlePattern] = None
    market_structures: List[MarketStructure] = None
    chart_patterns: List[ChartPattern] = None
    risk_reward: float = 0.0
    probability: float = 0.0  # 0-1 scale indicating setup probability
    ml_score: float = 0.0  # ML model score if available
    executed: bool = False
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.liquidity_zones is None:
            self.liquidity_zones = []
        if self.candle_patterns is None:
            self.candle_patterns = []
        if self.market_structures is None:
            self.market_structures = []
        if self.chart_patterns is None:
            self.chart_patterns = []
        
        # Calculate risk-reward ratio
        if self.direction == 1:  # Long
            self.risk_reward = (self.take_profit - self.entry_price) / (self.entry_price - self.stop_loss) if self.entry_price != self.stop_loss else 0
        else:  # Short
            self.risk_reward = (self.entry_price - self.take_profit) / (self.stop_loss - self.entry_price) if self.stop_loss != self.entry_price else 0
    
    def is_valid(self) -> bool:
        """Check if the trade setup is still valid."""
        if self.executed:
            return False
        
        if self.expiry_time and datetime.now() > self.expiry_time:
            return False
        
        # Minimum criteria for a valid setup
        if not self.liquidity_zones or self.risk_reward <= 0:
            return False
        
        # If ML scoring is used, check minimum score
        if self.ml_score > 0 and self.ml_score < 0.6:
            return False
        
        return True


class SMCStrategy(BaseStrategy):
    """Smart Money Concepts (SMC) trading strategy implementation."""
    
    def __init__(self, market_data: MarketData, order_manager: OrderManager, params: dict):
        """Initialize SMC strategy.
        
        Args:
            market_data: MarketData instance for price data
            order_manager: OrderManager instance for trade execution
            params: Strategy parameters
        """
        self.logger = logging.getLogger('main_logger')
        self.market_data = market_data
        self.order_manager = order_manager
        self.params = params
        
        # Store detected zones and patterns
        self.liquidity_zones = {}
        self.candle_patterns = {}
        self.market_structures = {}
        self.chart_patterns = {}
        self.trade_setups = []
    
    def analyze(self) -> List[TradeSetup]:
        """Run full strategy analysis and generate trade setups.
        
        Returns:
            List of valid trade setups
        """
        # Clear previous trade setups
        self.trade_setups = []
        
        # Analyze each symbol and timeframe
        for symbol in self.market_data.symbols:
            for timeframe in self.params.timeframes:
                # Get market data
                df = self.market_data.get_data(symbol, timeframe)
                if df is None or len(df) < self.params.lookback_period * 2:
                    continue
                
                # Run analysis components
                self._detect_liquidity_zones(symbol, timeframe, df)
                self._detect_candle_patterns(symbol, timeframe, df)
                self._analyze_market_structure(symbol, timeframe, df)
                
                if self.params.pattern_recognition_enabled:
                    self._detect_chart_patterns(symbol, timeframe, df)
                
                # Generate trade setups
                self._generate_trade_setups(symbol, timeframe, df)
        
        # Return only valid trade setups
        return [setup for setup in self.trade_setups if setup.is_valid()]
    
    def _detect_liquidity_zones(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect liquidity zones in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Get parameters
        lookback = self.params.lookback_period
        padding = self.params.liquidity_zone_padding
        
        # Initialize zones list for this symbol and timeframe
        key = f"{symbol}_{timeframe}"
        if key not in self.liquidity_zones:
            self.liquidity_zones[key] = []
        
        # Remove expired zones
        self.liquidity_zones[key] = [zone for zone in self.liquidity_zones[key] if zone.is_valid()]
        
        # 1. Detect swing highs and lows
        self._detect_swing_points(symbol, timeframe, df)
        
        # 2. Detect order blocks
        self._detect_order_blocks(symbol, timeframe, df)
        
        # 3. Detect Fair Value Gaps (FVG)
        self._detect_fair_value_gaps(symbol, timeframe, df)
        
        # 4. Detect consolidation zones
        self._detect_consolidation_zones(symbol, timeframe, df)
    
    def _detect_swing_points(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect swing highs and lows in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        lookback = self.params.lookback_period
        padding = self.params.liquidity_zone_padding
        key = f"{symbol}_{timeframe}"
        
        # We need at least 2*lookback+1 bars
        if len(df) < 2 * lookback + 1:
            return
        
        # Analyze recent price action (last 3*lookback bars)
        recent_df = df.iloc[-3*lookback:].copy()
        
        # Find swing highs
        for i in range(lookback, len(recent_df) - lookback):
            # Check if this is a local maximum
            if recent_df.iloc[i]['high'] == recent_df.iloc[i-lookback:i+lookback+1]['high'].max():
                # Calculate swing strength based on surrounding bars
                left_bars = recent_df.iloc[i-lookback:i]
                right_bars = recent_df.iloc[i+1:i+lookback+1]
                
                # Strength is based on how much higher this swing is compared to surrounding bars
                left_strength = (recent_df.iloc[i]['high'] - left_bars['high'].max()) / recent_df.iloc[i]['high'] if left_bars['high'].max() < recent_df.iloc[i]['high'] else 0
                right_strength = (recent_df.iloc[i]['high'] - right_bars['high'].max()) / recent_df.iloc[i]['high'] if right_bars['high'].max() < recent_df.iloc[i]['high'] else 0
                
                strength = (left_strength + right_strength) / 2
                
                # Only consider strong enough swings
                if strength >= self.params.swing_strength:
                    # Create liquidity zone
                    zone = LiquidityZone(
                        symbol=symbol,
                        timeframe=timeframe,
                        zone_type='swing_high',
                        price_high=recent_df.iloc[i]['high'] + padding,
                        price_low=recent_df.iloc[i]['high'] - padding,
                        strength=strength,
                        created_time=recent_df.index[i].to_pydatetime(),
                        # Expiry time is None (valid until swept)
                    )
                    
                    # Add to zones list
                    self.liquidity_zones[key].append(zone)
        
        # Find swing lows
        for i in range(lookback, len(recent_df) - lookback):
            # Check if this is a local minimum
            if recent_df.iloc[i]['low'] == recent_df.iloc[i-lookback:i+lookback+1]['low'].min():
                # Calculate swing strength based on surrounding bars
                left_bars = recent_df.iloc[i-lookback:i]
                right_bars = recent_df.iloc[i+1:i+lookback+1]
                
                # Strength is based on how much lower this swing is compared to surrounding bars
                left_strength = (left_bars['low'].min() - recent_df.iloc[i]['low']) / recent_df.iloc[i]['low'] if left_bars['low'].min() > recent_df.iloc[i]['low'] else 0
                right_strength = (right_bars['low'].min() - recent_df.iloc[i]['low']) / recent_df.iloc[i]['low'] if right_bars['low'].min() > recent_df.iloc[i]['low'] else 0
                
                strength = (left_strength + right_strength) / 2
                
                # Only consider strong enough swings
                if strength >= self.params.swing_strength:
                    # Create liquidity zone
                    zone = LiquidityZone(
                        symbol=symbol,
                        timeframe=timeframe,
                        zone_type='swing_low',
                        price_high=recent_df.iloc[i]['low'] + padding,
                        price_low=recent_df.iloc[i]['low'] - padding,
                        strength=strength,
                        created_time=recent_df.index[i].to_pydatetime(),
                        # Expiry time is None (valid until swept)
                    )
                    
                    # Add to zones list
                    self.liquidity_zones[key].append(zone)
    
    def _detect_order_blocks(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect order blocks in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        ob_candle_count = self.params.ob_candle_count
        min_body_size = self.params.ob_min_body_size
        imbalance_threshold = self.params.ob_imbalance_threshold
        key = f"{symbol}_{timeframe}"
        
        # We need at least ob_candle_count + 10 bars
        if len(df) < ob_candle_count + 10:
            return
        
        # Analyze recent price action (last 100 bars)
        recent_df = df.iloc[-100:].copy()
        
        # Find bullish order blocks (bearish candle followed by strong bullish move)
        for i in range(len(recent_df) - ob_candle_count - 1):
            # Check for bearish candle
            if recent_df.iloc[i]['direction'] == -1 and recent_df.iloc[i]['body_size'] >= min_body_size:
                # Check for bullish move in the next ob_candle_count candles
                next_candles = recent_df.iloc[i+1:i+ob_candle_count+1]
                if next_candles['direction'].sum() > 0 and next_candles['close'].iloc[-1] > recent_df.iloc[i]['high']:
                    # Calculate imbalance
                    price_move = next_candles['close'].iloc[-1] - recent_df.iloc[i]['close']
                    time_taken = ob_candle_count
                    imbalance = price_move / (time_taken * recent_df.iloc[i]['body_size'])
                    
                    if imbalance >= imbalance_threshold:
                        # Create order block zone
                        zone = LiquidityZone(
                            symbol=symbol,
                            timeframe=timeframe,
                            zone_type='order_block_bullish',
                            price_high=recent_df.iloc[i]['close'],
                            price_low=recent_df.iloc[i]['open'],
                            strength=min(1.0, imbalance / 2),  # Normalize strength
                            created_time=recent_df.index[i].to_pydatetime(),
                            # Expiry time is None (valid until swept)
                        )
                        
                        # Add to zones list
                        self.liquidity_zones[key].append(zone)
        
        # Find bearish order blocks (bullish candle followed by strong bearish move)
        for i in range(len(recent_df) - ob_candle_count - 1):
            # Check for bullish candle
            if recent_df.iloc[i]['direction'] == 1 and recent_df.iloc[i]['body_size'] >= min_body_size:
                # Check for bearish move in the next ob_candle_count candles
                next_candles = recent_df.iloc[i+1:i+ob_candle_count+1]
                if next_candles['direction'].sum() < 0 and next_candles['close'].iloc[-1] < recent_df.iloc[i]['low']:
                    # Calculate imbalance
                    price_move = recent_df.iloc[i]['close'] - next_candles['close'].iloc[-1]
                    time_taken = ob_candle_count
                    imbalance = price_move / (time_taken * recent_df.iloc[i]['body_size'])
                    
                    if imbalance >= imbalance_threshold:
                        # Create order block zone
                        zone = LiquidityZone(
                            symbol=symbol,
                            timeframe=timeframe,
                            zone_type='order_block_bearish',
                            price_high=recent_df.iloc[i]['open'],
                            price_low=recent_df.iloc[i]['close'],
                            strength=min(1.0, imbalance / 2),  # Normalize strength
                            created_time=recent_df.index[i].to_pydatetime(),
                            # Expiry time is None (valid until swept)
                        )
                        
                        # Add to zones list
                        self.liquidity_zones[key].append(zone)
    
    def _detect_fair_value_gaps(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect Fair Value Gaps (FVG) in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        min_gap_size = self.params.fvg_min_gap_size
        max_candles = self.params.fvg_max_candles
        key = f"{symbol}_{timeframe}"
        
        # We need at least 3 bars
        if len(df) < 3:
            return
        
        # Analyze recent price action (last 100 bars)
        recent_df = df.iloc[-100:].copy()
        
        # Find bullish FVG (gap up)
        for i in range(len(recent_df) - 2):
            # Check for gap between candle i and i+2
            if recent_df.iloc[i+2]['low'] > recent_df.iloc[i]['high']:
                gap_size = recent_df.iloc[i+2]['low'] - recent_df.iloc[i]['high']
                
                if gap_size >= min_gap_size:
                    # Create FVG zone
                    zone = LiquidityZone(
                        symbol=symbol,
                        timeframe=timeframe,
                        zone_type='fvg_bullish',
                        price_high=recent_df.iloc[i+2]['low'],
                        price_low=recent_df.iloc[i]['high'],
                        strength=min(1.0, gap_size / (min_gap_size * 5)),  # Normalize strength
                        created_time=recent_df.index[i+1].to_pydatetime(),
                        expiry_time=recent_df.index[i+1].to_pydatetime() + pd.Timedelta(max_candles, unit='h'),
                    )
                    
                    # Add to zones list
                    self.liquidity_zones[key].append(zone)
        
        # Find bearish FVG (gap down)
        for i in range(len(recent_df) - 2):
            # Check for gap between candle i and i+2
            if recent_df.iloc[i+2]['high'] < recent_df.iloc[i]['low']:
                gap_size = recent_df.iloc[i]['low'] - recent_df.iloc[i+2]['high']
                
                if gap_size >= min_gap_size:
                    # Create FVG zone
                    zone = LiquidityZone(
                        symbol=symbol,
                        timeframe=timeframe,
                        zone_type='fvg_bearish',
                        price_high=recent_df.iloc[i]['low'],
                        price_low=recent_df.iloc[i+2]['high'],
                        strength=min(1.0, gap_size / (min_gap_size * 5)),  # Normalize strength
                        created_time=recent_df.index[i+1].to_pydatetime(),
                        expiry_time=recent_df.index[i+1].to_pydatetime() + pd.Timedelta(max_candles, unit='h'),
                    )
                    
                    # Add to zones list
                    self.liquidity_zones[key].append(zone)
    
    def _detect_consolidation_zones(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect consolidation zones (liquidity pools) in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        range_threshold = self.params.consolidation_range_threshold
        min_candles = self.params.consolidation_min_candles
        key = f"{symbol}_{timeframe}"
        
        # We need at least min_candles bars
        if len(df) < min_candles:
            return
        
        # Analyze recent price action (last 100 bars)
        recent_df = df.iloc[-100:].copy()
        
        # Detect consolidation zones using rolling window
        for i in range(len(recent_df) - min_candles + 1):
            window = recent_df.iloc[i:i+min_candles]
            price_range = window['high'].max() - window['low'].min()
            avg_candle_size = window['candle_size'].mean()
            
            # Check if price range is small enough relative to average candle size
            if price_range <= range_threshold:
                # Create consolidation zone
                zone = LiquidityZone(
                    symbol=symbol,
                    timeframe=timeframe,
                    zone_type='consolidation',
                    price_high=window['high'].max(),
                    price_low=window['low'].min(),
                    strength=1.0 - (price_range / range_threshold),  # Higher strength for tighter ranges
                    created_time=window.index[0].to_pydatetime(),
                    expiry_time=window.index[-1].to_pydatetime() + pd.Timedelta(min_candles * 2, unit='h'),
                )
                
                # Add to zones list
                self.liquidity_zones[key].append(zone)
    
    def _detect_candle_patterns(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect candlestick patterns in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Initialize patterns list for this symbol and timeframe
        key = f"{symbol}_{timeframe}"
        if key not in self.candle_patterns:
            self.candle_patterns[key] = []
        
        # Clear old patterns (only keep recent ones)
        self.candle_patterns[key] = []
        
        # We need at least 3 bars
        if len(df) < 3:
            return
        
        # Analyze recent price action (last 20 bars)
        recent_df = df.iloc[-20:].copy()
        
        # 1. Detect engulfing patterns
        self._detect_engulfing_patterns(symbol, timeframe, recent_df)
        
        # 2. Detect pin bars
        self._detect_pin_bars(symbol, timeframe, recent_df)
        
        # 3. Detect inside bars
        self._detect_inside_bars(symbol, timeframe, recent_df)
        
        # 4. Detect doji
        self._detect_doji(symbol, timeframe, recent_df)
    
    def _detect_engulfing_patterns(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect engulfing candlestick patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        engulfing_ratio = self.params.engulfing_body_ratio
        key = f"{symbol}_{timeframe}"
        
        # We need at least 2 bars
        if len(df) < 2:
            return
        
        # Check each pair of consecutive candles
        for i in range(1, len(df)):
            prev_candle = df.iloc[i-1]
            curr_candle = df.iloc[i]
            
            # Bullish engulfing
            if (curr_candle['direction'] == 1 and  # Current candle is bullish
                prev_candle['direction'] == -1 and  # Previous candle is bearish
                curr_candle['body_size'] > prev_candle['body_size'] * engulfing_ratio and  # Current body is larger
                curr_candle['open'] <= prev_candle['close'] and  # Current open below previous close
                curr_candle['close'] >= prev_candle['open']):  # Current close above previous open
                
                # Calculate pattern strength
                strength = min(1.0, curr_candle['body_size'] / (prev_candle['body_size'] * engulfing_ratio * 2))
                
                # Create pattern
                pattern = CandlePattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='engulfing_bullish',
                    bar_index=i,
                    strength=strength,
                    created_time=df.index[i].to_pydatetime()
                )
                
                # Add to patterns list
                self.candle_patterns[key].append(pattern)
            
            # Bearish engulfing
            elif (curr_candle['direction'] == -1 and  # Current candle is bearish
                  prev_candle['direction'] == 1 and  # Previous candle is bullish
                  curr_candle['body_size'] > prev_candle['body_size'] * engulfing_ratio and  # Current body is larger
                  curr_candle['open'] >= prev_candle['close'] and  # Current open above previous close
                  curr_candle['close'] <= prev_candle['open']):  # Current close below previous open
                
                # Calculate pattern strength
                strength = min(1.0, curr_candle['body_size'] / (prev_candle['body_size'] * engulfing_ratio * 2))
                
                # Create pattern
                pattern = CandlePattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='engulfing_bearish',
                    bar_index=i,
                    strength=strength,
                    created_time=df.index[i].to_pydatetime()
                )
                
                # Add to patterns list
                self.candle_patterns[key].append(pattern)
    
    def _detect_pin_bars(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect pin bar (hammer/shooting star) candlestick patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        nose_ratio = self.params.pin_bar_nose_ratio
        key = f"{symbol}_{timeframe}"
        
        # Check each candle
        for i in range(len(df)):
            candle = df.iloc[i]
            
            # Bullish pin bar (hammer)
            if (candle['lower_wick'] > candle['body_size'] * 2 and  # Long lower wick
                candle['lower_wick'] > candle['upper_wick'] * 3 and  # Lower wick much longer than upper
                candle['lower_wick'] / candle['candle_size'] > nose_ratio):  # Lower wick is significant portion
                
                # Calculate pattern strength
                strength = min(1.0, candle['lower_wick'] / (candle['body_size'] * 3))
                
                # Create pattern
                pattern = CandlePattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='pin_bar_bullish',
                    bar_index=i,
                    strength=strength,
                    created_time=df.index[i].to_pydatetime()
                )
                
                # Add to patterns list
                self.candle_patterns[key].append(pattern)
            
            # Bearish pin bar (shooting star)
            elif (candle['upper_wick'] > candle['body_size'] * 2 and  # Long upper wick
                  candle['upper_wick'] > candle['lower_wick'] * 3 and  # Upper wick much longer than lower
                  candle['upper_wick'] / candle['candle_size'] > nose_ratio):  # Upper wick is significant portion
                
                # Calculate pattern strength
                strength = min(1.0, candle['upper_wick'] / (candle['body_size'] * 3))
                
                # Create pattern
                pattern = CandlePattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='pin_bar_bearish',
                    bar_index=i,
                    strength=strength,
                    created_time=df.index[i].to_pydatetime()
                )
                
                # Add to patterns list
                self.candle_patterns[key].append(pattern)
    
    def _detect_inside_bars(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect inside bar candlestick patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        key = f"{symbol}_{timeframe}"
        
        # We need at least 2 bars
        if len(df) < 2:
            return
        
        # Check each pair of consecutive candles
        for i in range(1, len(df)):
            prev_candle = df.iloc[i-1]
            curr_candle = df.iloc[i]
            
            # Inside bar
            if (curr_candle['high'] < prev_candle['high'] and  # Current high below previous high
                curr_candle['low'] > prev_candle['low']):  # Current low above previous low
                
                # Calculate pattern strength based on how much smaller the inside bar is
                size_ratio = curr_candle['candle_size'] / prev_candle['candle_size']
                strength = min(1.0, 1.0 - size_ratio)  # Smaller inside bar = stronger pattern
                
                # Create pattern
                pattern = CandlePattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='inside_bar',
                    bar_index=i,
                    strength=strength,
                    created_time=df.index[i].to_pydatetime()
                )
                
                # Add to patterns list
                self.candle_patterns[key].append(pattern)
    
    def _detect_doji(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect doji candlestick patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        doji_threshold = self.params.doji_body_threshold
        key = f"{symbol}_{timeframe}"
        
        # Check each candle
        for i in range(len(df)):
            candle = df.iloc[i]
            
            # Doji has very small body compared to total candle size
            if candle['body_size'] <= doji_threshold and candle['candle_size'] > doji_threshold * 3:
                # Calculate pattern strength
                body_to_candle_ratio = candle['body_size'] / candle['candle_size']
                strength = min(1.0, 1.0 - (body_to_candle_ratio * 10))  # Smaller body = stronger doji
                
                # Create pattern
                pattern = CandlePattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='doji',
                    bar_index=i,
                    strength=strength,
                    created_time=df.index[i].to_pydatetime()
                )
                
                # Add to patterns list
                self.candle_patterns[key].append(pattern)
    
    def _analyze_market_structure(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Analyze market structure (BOS, CHoCH, HH/HL, LH/LL).
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Initialize structures list for this symbol and timeframe
        key = f"{symbol}_{timeframe}"
        if key not in self.market_structures:
            self.market_structures[key] = []
        
        # Clear old structures (only keep recent ones)
        self.market_structures[key] = []
        
        # We need enough bars for structure analysis
        lookback = self.params.structure_swing_lookback
        if len(df) < lookback * 3:
            return
        
        # Analyze recent price action
        recent_df = df.iloc[-lookback*3:].copy()
        
        # Find swing highs and lows for structure analysis
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(recent_df) - lookback):
            # Check if this is a local maximum
            if recent_df.iloc[i]['high'] == recent_df.iloc[i-lookback:i+lookback+1]['high'].max():
                swing_highs.append((i, recent_df.iloc[i]['high'], recent_df.index[i]))
            
            # Check if this is a local minimum
            if recent_df.iloc[i]['low'] == recent_df.iloc[i-lookback:i+lookback+1]['low'].min():
                swing_lows.append((i, recent_df.iloc[i]['low'], recent_df.index[i]))
        
        # Need at least 2 swing points of each type
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return
        
        # Analyze higher highs / lower highs
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs[i-1][1]
            curr_high = swing_highs[i][1]
            
            if curr_high > prev_high:  # Higher high
                # Create structure
                structure = MarketStructure(
                    symbol=symbol,
                    timeframe=timeframe,
                    structure_type='hh_hl',  # Higher high
                    price_level=curr_high,
                    created_time=swing_highs[i][2].to_pydatetime(),
                    strength=min(1.0, (curr_high - prev_high) / prev_high * 100)  # Strength based on % increase
                )
                
                # Add to structures list
                self.market_structures[key].append(structure)
            elif curr_high < prev_high:  # Lower high
                # Create structure
                structure = MarketStructure(
                    symbol=symbol,
                    timeframe=timeframe,
                    structure_type='ll_lh',  # Lower high
                    price_level=curr_high,
                    created_time=swing_highs[i][2].to_pydatetime(),
                    strength=min(1.0, (prev_high - curr_high) / prev_high * 100)  # Strength based on % decrease
                )
                
                # Add to structures list
                self.market_structures[key].append(structure)
        
        # Analyze higher lows / lower lows
        for i in range(1, len(swing_lows)):
            prev_low = swing_lows[i-1][1]
            curr_low = swing_lows[i][1]
            
            if curr_low > prev_low:  # Higher low
                # Create structure
                structure = MarketStructure(
                    symbol=symbol,
                    timeframe=timeframe,
                    structure_type='hh_hl',  # Higher low
                    price_level=curr_low,
                    created_time=swing_lows[i][2].to_pydatetime(),
                    strength=min(1.0, (curr_low - prev_low) / prev_low * 100)  # Strength based on % increase
                )
                
                # Add to structures list
                self.market_structures[key].append(structure)
            elif curr_low < prev_low:  # Lower low
                # Create structure
                structure = MarketStructure(
                    symbol=symbol,
                    timeframe=timeframe,
                    structure_type='ll_lh',  # Lower low
                    price_level=curr_low,
                    created_time=swing_lows[i][2].to_pydatetime(),
                    strength=min(1.0, (prev_low - curr_low) / prev_low * 100)  # Strength based on % decrease
                )
                
                # Add to structures list
                self.market_structures[key].append(structure)
        
        # Detect Break of Structure (BOS)
        self._detect_break_of_structure(symbol, timeframe, recent_df, swing_highs, swing_lows)
        
        # Detect Change of Character (CHoCH)
        self._detect_change_of_character(symbol, timeframe, recent_df, swing_highs, swing_lows)
    
    def _detect_break_of_structure(self, symbol: str, timeframe: int, df: pd.DataFrame, 
                                  swing_highs: list, swing_lows: list) -> None:
        """Detect Break of Structure (BOS) in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
            swing_highs: List of swing high points (index, price, time)
            swing_lows: List of swing low points (index, price, time)
        """
        confirmation_candles = self.params.bos_confirmation_candles
        key = f"{symbol}_{timeframe}"
        
        # Need at least 2 swing points of each type
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return
        
        # Get the most recent candle
        last_candle_idx = len(df) - 1
        last_candle = df.iloc[last_candle_idx]
        
        # Check for bullish BOS (price breaks above recent swing high)
        for i in range(len(swing_highs) - 1, 0, -1):  # Start from most recent
            swing_idx, swing_price, swing_time = swing_highs[i]
            
            # Check if price has broken above this swing high
            if last_candle['close'] > swing_price:
                # Check if we have confirmation (price stayed above for confirmation_candles)
                confirmed = True
                for j in range(1, min(confirmation_candles + 1, last_candle_idx - swing_idx)):
                    if df.iloc[swing_idx + j]['close'] < swing_price:
                        confirmed = False
                        break
                
                if confirmed:
                    # Calculate strength based on how far price has moved above the swing high
                    breakout_size = (last_candle['close'] - swing_price) / swing_price
                    strength = min(1.0, breakout_size * 100)  # Normalize strength
                    
                    # Create structure
                    structure = MarketStructure(
                        symbol=symbol,
                        timeframe=timeframe,
                        structure_type='bos_bullish',
                        price_level=swing_price,
                        created_time=df.index[last_candle_idx].to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to structures list
                    self.market_structures[key].append(structure)
                    break  # Only consider the most recent break
        
        # Check for bearish BOS (price breaks below recent swing low)
        for i in range(len(swing_lows) - 1, 0, -1):  # Start from most recent
            swing_idx, swing_price, swing_time = swing_lows[i]
            
            # Check if price has broken below this swing low
            if last_candle['close'] < swing_price:
                # Check if we have confirmation (price stayed below for confirmation_candles)
                confirmed = True
                for j in range(1, min(confirmation_candles + 1, last_candle_idx - swing_idx)):
                    if df.iloc[swing_idx + j]['close'] > swing_price:
                        confirmed = False
                        break
                
                if confirmed:
                    # Calculate strength based on how far price has moved below the swing low
                    breakout_size = (swing_price - last_candle['close']) / swing_price
                    strength = min(1.0, breakout_size * 100)  # Normalize strength
                    
                    # Create structure
                    structure = MarketStructure(
                        symbol=symbol,
                        timeframe=timeframe,
                        structure_type='bos_bearish',
                        price_level=swing_price,
                        created_time=df.index[last_candle_idx].to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to structures list
                    self.market_structures[key].append(structure)
                    break  # Only consider the most recent break
    
    def _detect_change_of_character(self, symbol: str, timeframe: int, df: pd.DataFrame, 
                                   swing_highs: list, swing_lows: list) -> None:
        """Detect Change of Character (CHoCH) in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
            swing_highs: List of swing high points (index, price, time)
            swing_lows: List of swing low points (index, price, time)
        """
        key = f"{symbol}_{timeframe}"
        
        # Need at least 3 swing points of each type
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return
        
        # Check for bullish CHoCH (higher low after lower low)
        for i in range(len(swing_lows) - 1, 1, -1):  # Start from most recent
            curr_idx, curr_price, curr_time = swing_lows[i]
            prev_idx, prev_price, prev_time = swing_lows[i-1]
            prev_prev_idx, prev_prev_price, prev_prev_time = swing_lows[i-2]
            
            # Check for pattern: lower low followed by higher low
            if prev_price < prev_prev_price and curr_price > prev_price:
                # Calculate strength based on the size of the reversal
                reversal_size = (curr_price - prev_price) / prev_price
                strength = min(1.0, reversal_size * 100)  # Normalize strength
                
                # Create structure
                structure = MarketStructure(
                    symbol=symbol,
                    timeframe=timeframe,
                    structure_type='choch_bullish',
                    price_level=curr_price,
                    created_time=curr_time.to_pydatetime(),
                    strength=strength
                )
                
                # Add to structures list
                self.market_structures[key].append(structure)
                break  # Only consider the most recent CHoCH
        
        # Check for bearish CHoCH (lower high after higher high)
        for i in range(len(swing_highs) - 1, 1, -1):  # Start from most recent
            curr_idx, curr_price, curr_time = swing_highs[i]
            prev_idx, prev_price, prev_time = swing_highs[i-1]
            prev_prev_idx, prev_prev_price, prev_prev_time = swing_highs[i-2]
            
            # Check for pattern: higher high followed by lower high
            if prev_price > prev_prev_price and curr_price < prev_price:
                # Calculate strength based on the size of the reversal
                reversal_size = (prev_price - curr_price) / prev_price
                strength = min(1.0, reversal_size * 100)  # Normalize strength
                
                # Create structure
                structure = MarketStructure(
                    symbol=symbol,
                    timeframe=timeframe,
                    structure_type='choch_bearish',
                    price_level=curr_price,
                    created_time=curr_time.to_pydatetime(),
                    strength=strength
                )
                
                # Add to structures list
                self.market_structures[key].append(structure)
                break  # Only consider the most recent CHoCH
    
    def _detect_chart_patterns(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect chart patterns in the market data.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Initialize patterns list for this symbol and timeframe
        key = f"{symbol}_{timeframe}"
        if key not in self.chart_patterns:
            self.chart_patterns[key] = []
        
        # Clear old patterns (only keep recent ones)
        self.chart_patterns[key] = []
        
        # We need enough bars for pattern detection
        min_candles = self.params.pattern_min_candles
        max_candles = self.params.pattern_max_candles
        if len(df) < max_candles:
            return
        
        # Analyze recent price action
        recent_df = df.iloc[-max_candles:].copy()
        
        # Detect double tops and bottoms
        self._detect_double_patterns(symbol, timeframe, recent_df)
        
        # Detect head and shoulders patterns
        self._detect_head_shoulders(symbol, timeframe, recent_df)
        
        # Detect triangle patterns
        self._detect_triangles(symbol, timeframe, recent_df)
        
        # Detect flag patterns
        self._detect_flags(symbol, timeframe, recent_df)
    
    def _detect_double_patterns(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect double top and double bottom patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Implementation for double top/bottom detection
        # This is a simplified version - in a real implementation, you would use more sophisticated algorithms
        key = f"{symbol}_{timeframe}"
        lookback = self.params.structure_swing_lookback
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check if this is a local maximum
            if df.iloc[i]['high'] == df.iloc[i-lookback:i+lookback+1]['high'].max():
                swing_highs.append((i, df.iloc[i]['high'], df.index[i]))
            
            # Check if this is a local minimum
            if df.iloc[i]['low'] == df.iloc[i-lookback:i+lookback+1]['low'].min():
                swing_lows.append((i, df.iloc[i]['low'], df.index[i]))
        
        # Need at least 2 swing points of each type
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return
        
        # Check for double tops
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                idx1, price1, time1 = swing_highs[i]
                idx2, price2, time2 = swing_highs[j]
                
                # Check if prices are similar (within 0.5%)
                if abs(price1 - price2) / price1 < 0.005 and idx2 - idx1 > lookback * 2:
                    # Find the lowest point between the two tops
                    between_df = df.iloc[idx1:idx2+1]
                    neckline = between_df['low'].min()
                    
                    # Calculate pattern strength
                    height = price1 - neckline
                    strength = min(1.0, height / price1 * 20)  # Normalize strength
                    
                    # Calculate target price (equal to height)
                    target = neckline - height
                    
                    # Create pattern
                    pattern = ChartPattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type='double_top',
                        price_high=max(price1, price2),
                        price_low=neckline,
                        target_price=target,
                        created_time=time2.to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to patterns list
                    self.chart_patterns[key].append(pattern)
        
        # Check for double bottoms
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                idx1, price1, time1 = swing_lows[i]
                idx2, price2, time2 = swing_lows[j]
                
                # Check if prices are similar (within 0.5%)
                if abs(price1 - price2) / price1 < 0.005 and idx2 - idx1 > lookback * 2:
                    # Find the highest point between the two bottoms
                    between_df = df.iloc[idx1:idx2+1]
                    neckline = between_df['high'].max()
                    
                    # Calculate pattern strength
                    height = neckline - price1
                    strength = min(1.0, height / price1 * 20)  # Normalize strength
                    
                    # Calculate target price (equal to height)
                    target = neckline + height
                    
                    # Create pattern
                    pattern = ChartPattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type='double_bottom',
                        price_high=neckline,
                        price_low=min(price1, price2),
                        target_price=target,
                        created_time=time2.to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to patterns list
                    self.chart_patterns[key].append(pattern)
    
    def _detect_head_shoulders(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect head and shoulders patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Implementation for head and shoulders detection
        # This is a simplified version - in a real implementation, you would use more sophisticated algorithms
        key = f"{symbol}_{timeframe}"
        lookback = self.params.structure_swing_lookback
        
        # Find swing highs and lows
        swing_highs = []
        
        for i in range(lookback, len(df) - lookback):
            # Check if this is a local maximum
            if df.iloc[i]['high'] == df.iloc[i-lookback:i+lookback+1]['high'].max():
                swing_highs.append((i, df.iloc[i]['high'], df.index[i]))
        
        # Need at least 3 swing highs for head and shoulders
        if len(swing_highs) < 3:
            return
        
        # Check for head and shoulders pattern
        for i in range(len(swing_highs) - 2):
            left_idx, left_price, left_time = swing_highs[i]
            head_idx, head_price, head_time = swing_highs[i+1]
            right_idx, right_price, right_time = swing_highs[i+2]
            
            # Check if head is higher than shoulders
            if head_price > left_price and head_price > right_price:
                # Check if shoulders are at similar levels (within 1%)
                if abs(left_price - right_price) / left_price < 0.01:
                    # Find the neckline (connecting the lows between shoulders and head)
                    left_low = df.iloc[left_idx:head_idx]['low'].min()
                    right_low = df.iloc[head_idx:right_idx]['low'].min()
                    neckline = (left_low + right_low) / 2
                    
                    # Calculate pattern strength
                    height = head_price - neckline
                    strength = min(1.0, height / head_price * 20)  # Normalize strength
                    
                    # Calculate target price (equal to height)
                    target = neckline - height
                    
                    # Create pattern
                    pattern = ChartPattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type='head_shoulders',
                        price_high=head_price,
                        price_low=neckline,
                        target_price=target,
                        created_time=right_time.to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to patterns list
                    self.chart_patterns[key].append(pattern)
    
    def _detect_triangles(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect triangle patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Implementation for triangle pattern detection
        # This is a simplified version - in a real implementation, you would use more sophisticated algorithms
        key = f"{symbol}_{timeframe}"
        min_points = 5  # Minimum points to form a triangle
        
        # Need enough data points
        if len(df) < min_points * 2:
            return
        
        # Check for ascending triangle (flat top, rising bottom)
        # Find potential resistance level (flat top)
        highs = df['high'].values
        resistance_level = None
        resistance_count = 0
        
        for i in range(len(highs) - 1):
            level = highs[i]
            count = 0
            
            for j in range(i + 1, len(highs)):
                if abs(highs[j] - level) / level < 0.002:
                    count += 1
            
            if count >= min_points and (resistance_level is None or count > resistance_count):
                resistance_level = level
                resistance_count = count
        
        # Find rising support line (bottom)
        if resistance_level is not None and resistance_count >= min_points:
            # Get the lows
            lows = df['low'].values
            
            # Find points that form the rising bottom
            bottom_points = []
            
            for i in range(len(lows) - 1):
                is_low = True
                
                for j in range(max(0, i - 3), i):
                    if lows[j] <= lows[i]:
                        is_low = False
                        break
                
                if is_low:
                    bottom_points.append((i, lows[i]))
            
            # Need at least 3 points to form a rising bottom
            if len(bottom_points) >= 3:
                # Check if points are rising
                rising = True
                for i in range(1, len(bottom_points)):
                    if bottom_points[i][1] <= bottom_points[i-1][1]:
                        rising = False
                        break
                
                if rising:
                    # Calculate pattern strength
                    height = resistance_level - bottom_points[0][1]
                    strength = min(1.0, height / resistance_level * 20)  # Normalize strength
                    
                    # Calculate target price (equal to height)
                    target = resistance_level + height
                    
                    # Create pattern
                    pattern = ChartPattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type='triangle',
                        price_high=resistance_level,
                        price_low=bottom_points[0][1],
                        target_price=target,
                        created_time=df.index[-1].to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to patterns list
                    self.chart_patterns[key].append(pattern)
    
    def _detect_flags(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Detect flag patterns.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        # Implementation for flag pattern detection
        # This is a simplified version - in a real implementation, you would use more sophisticated algorithms
        key = f"{symbol}_{timeframe}"
        min_pole_size = self.params.flag_min_pole_size
        
        # Need enough data points
        if len(df) < 20:
            return
        
        # Check for bullish flag
        # First, find a strong upward move (pole)
        for i in range(len(df) - 10):
            # Check for a strong bullish candle or series of candles
            pole_start = i
            pole_end = i
            
            for j in range(i, min(i + 5, len(df) - 5)):
                if df.iloc[j]['direction'] == 1 and df.iloc[j]['body_size'] > min_pole_size:
                    pole_end = j
                else:
                    break
            
            # If we found a pole
            if pole_end > pole_start:
                pole_height = df.iloc[pole_end]['high'] - df.iloc[pole_start]['low']
                
                # Check for consolidation after the pole
                flag_start = pole_end + 1
                flag_end = flag_start
                
                for j in range(flag_start, min(flag_start + 10, len(df))):
                    # Flag should be a consolidation or small pullback
                    if df.iloc[j]['candle_size'] < pole_height * 0.3:
                        flag_end = j
                    else:
                        break
                
                # If we found a flag
                if flag_end > flag_start + 2:  # At least 3 candles in the flag
                    # Calculate pattern strength
                    strength = min(1.0, pole_height / df.iloc[pole_start]['low'] * 10)  # Normalize strength
                    
                    # Calculate target price (equal to pole height)
                    target = df.iloc[flag_end]['high'] + pole_height
                    
                    # Create pattern
                    pattern = ChartPattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type='flag_bullish',
                        price_high=df.iloc[flag_end]['high'],
                        price_low=df.iloc[pole_start]['low'],
                        target_price=target,
                        created_time=df.index[flag_end].to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to patterns list
                    self.chart_patterns[key].append(pattern)
        
        # Check for bearish flag
        # First, find a strong downward move (pole)
        for i in range(len(df) - 10):
            # Check for a strong bearish candle or series of candles
            pole_start = i
            pole_end = i
            
            for j in range(i, min(i + 5, len(df) - 5)):
                if df.iloc[j]['direction'] == -1 and df.iloc[j]['body_size'] > min_pole_size:
                    pole_end = j
                else:
                    break
            
            # If we found a pole
            if pole_end > pole_start:
                pole_height = df.iloc[pole_start]['high'] - df.iloc[pole_end]['low']
                
                # Check for consolidation after the pole
                flag_start = pole_end + 1
                flag_end = flag_start
                
                for j in range(flag_start, min(flag_start + 10, len(df))):
                    # Flag should be a consolidation or small pullback
                    if df.iloc[j]['candle_size'] < pole_height * 0.3:
                        flag_end = j
                    else:
                        break
                
                # If we found a flag
                if flag_end > flag_start + 2:  # At least 3 candles in the flag
                    # Calculate pattern strength
                    strength = min(1.0, pole_height / df.iloc[pole_end]['low'] * 10)  # Normalize strength
                    
                    # Calculate target price (equal to pole height)
                    target = df.iloc[flag_end]['low'] - pole_height
                    
                    # Create pattern
                    pattern = ChartPattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type='flag_bearish',
                        price_high=df.iloc[pole_start]['high'],
                        price_low=df.iloc[flag_end]['low'],
                        target_price=target,
                        created_time=df.index[flag_end].to_pydatetime(),
                        strength=strength
                    )
                    
                    # Add to patterns list
                    self.chart_patterns[key].append(pattern)
    
    def _generate_trade_setups(self, symbol: str, timeframe: int, df: pd.DataFrame) -> None:
        """Generate trade setups based on analysis.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
        """
        key = f"{symbol}_{timeframe}"
        
        # Get current price
        current_price = df.iloc[-1]['close']
        
        # Get ATR for stop loss calculation
        atr = self.market_data.calculate_atr(symbol, timeframe, self.params.atr_period)
        
        # 1. Generate long setups
        self._generate_long_setups(symbol, timeframe, df, current_price, atr)
        
        # 2. Generate short setups
        self._generate_short_setups(symbol, timeframe, df, current_price, atr)
    
    def _generate_long_setups(self, symbol: str, timeframe: int, df: pd.DataFrame, 
                             current_price: float, atr: float) -> None:
        """Generate long trade setups.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
            current_price: Current price
            atr: Average True Range
        """
        key = f"{symbol}_{timeframe}"
        
        # Check if we have liquidity zones
        if key not in self.liquidity_zones or not self.liquidity_zones[key]:
            return
        
        # Get bullish liquidity zones
        bullish_zones = [zone for zone in self.liquidity_zones[key] 
                        if zone.is_bullish() and zone.is_valid()]
        
        # Get bullish market structures
        bullish_structures = []
        if key in self.market_structures:
            bullish_structures = [structure for structure in self.market_structures[key] 
                                if structure.is_bullish()]
        
        # Get bullish candle patterns
        bullish_patterns = []
        if key in self.candle_patterns:
            bullish_patterns = [pattern for pattern in self.candle_patterns[key] 
                              if pattern.is_bullish()]
        
        # Get bullish chart patterns
        bullish_chart_patterns = []
        if key in self.chart_patterns:
            bullish_chart_patterns = [pattern for pattern in self.chart_patterns[key] 
                                    if pattern.is_bullish()]
        
        # Generate setups for each bullish liquidity zone
        for zone in bullish_zones:
            # Skip zones that are too far from current price
            if abs(current_price - zone.mid_price) / current_price > self.params.max_zone_distance:
                continue
            
            # Calculate entry, stop loss, and take profit
            entry_price = zone.mid_price
            stop_loss = zone.price_low - atr * self.params.sl_atr_multiplier
            take_profit = entry_price + (entry_price - stop_loss) * self.params.rr_ratio
            
            # Create trade setup
            setup = TradeSetup(
                symbol=symbol,
                timeframe=timeframe,
                direction=1,  # Long
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_time=datetime.now(),
                expiry_time=datetime.now() + pd.Timedelta(self.params.setup_expiry_hours, unit='h'),
                liquidity_zones=[zone],
                candle_patterns=[p for p in bullish_patterns if p.is_bullish()],
                market_structures=[s for s in bullish_structures if s.is_bullish()],
                chart_patterns=[p for p in bullish_chart_patterns if p.is_bullish()],
            )
            
            # Calculate probability based on confluence
            self._calculate_setup_probability(setup)
            
            # Add to trade setups if probability is high enough
            if setup.probability >= self.params.min_setup_probability:
                self.trade_setups.append(setup)
    
    def _generate_short_setups(self, symbol: str, timeframe: int, df: pd.DataFrame, 
                              current_price: float, atr: float) -> None:
        """Generate short trade setups.
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            df: DataFrame with market data
            current_price: Current price
            atr: Average True Range
        """
        key = f"{symbol}_{timeframe}"
        
        # Check if we have liquidity zones
        if key not in self.liquidity_zones or not self.liquidity_zones[key]:
            return
        
        # Get bearish liquidity zones
        bearish_zones = [zone for zone in self.liquidity_zones[key] 
                        if zone.is_bearish() and zone.is_valid()]
        
        # Get bearish market structures
        bearish_structures = []
        if key in self.market_structures:
            bearish_structures = [structure for structure in self.market_structures[key] 
                                if structure.is_bearish()]
        
        # Get bearish candle patterns
        bearish_patterns = []
        if key in self.candle_patterns:
            bearish_patterns = [pattern for pattern in self.candle_patterns[key] 
                              if pattern.is_bearish()]
        
        # Get bearish chart patterns
        bearish_chart_patterns = []
        if key in self.chart_patterns:
            bearish_chart_patterns = [pattern for pattern in self.chart_patterns[key] 
                                    if pattern.is_bearish()]
        
        # Generate setups for each bearish liquidity zone
        for zone in bearish_zones:
            # Skip zones that are too far from current price
            if abs(current_price - zone.mid_price) / current_price > self.params.max_zone_distance:
                continue
            
            # Calculate entry, stop loss, and take profit
            entry_price = zone.mid_price
            stop_loss = zone.price_high + atr * self.params.sl_atr_multiplier
            take_profit = entry_price - (stop_loss - entry_price) * self.params.rr_ratio
            
            # Create trade setup
            setup = TradeSetup(
                symbol=symbol,
                timeframe=timeframe,
                direction=-1,  # Short
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_time=datetime.now(),
                expiry_time=datetime.now() + pd.Timedelta(self.params.setup_expiry_hours, unit='h'),
                liquidity_zones=[zone],
                candle_patterns=[p for p in bearish_patterns if p.is_bearish()],
                market_structures=[s for s in bearish_structures if s.is_bearish()],
                chart_patterns=[p for p in bearish_chart_patterns if p.is_bearish()],
            )
            
            # Calculate probability based on confluence
            self._calculate_setup_probability(setup)
            
            # Add to trade setups if probability is high enough
            if setup.probability >= self.params.min_setup_probability:
                self.trade_setups.append(setup)
    
    def _calculate_setup_probability(self, setup: TradeSetup) -> None:
        """Calculate the probability of a trade setup based on confluence.
        
        Args:
            setup: TradeSetup instance to calculate probability for
        """
        # Base probability
        probability = 0.0
        
        # 1. Liquidity zone strength
        if setup.liquidity_zones:
            avg_zone_strength = sum(zone.strength for zone in setup.liquidity_zones) / len(setup.liquidity_zones)
            probability += avg_zone_strength * self.params.zone_weight
        
        # 2. Candle pattern strength
        if setup.candle_patterns:
            avg_pattern_strength = sum(pattern.strength for pattern in setup.candle_patterns) / len(setup.candle_patterns)
            probability += avg_pattern_strength * self.params.candle_pattern_weight
        
        # 3. Market structure strength
        if setup.market_structures:
            avg_structure_strength = sum(structure.strength for structure in setup.market_structures) / len(setup.market_structures)
            probability += avg_structure_strength * self.params.structure_weight
        
        # 4. Chart pattern strength
        if setup.chart_patterns:
            avg_chart_pattern_strength = sum(pattern.strength for pattern in setup.chart_patterns) / len(setup.chart_patterns)
            probability += avg_chart_pattern_strength * self.params.chart_pattern_weight
        
        # 5. Risk-reward ratio
        if setup.risk_reward >= self.params.min_rr:
            probability += min(0.2, (setup.risk_reward - self.params.min_rr) / 10)
        
        # 6. ML score if available
        if setup.ml_score > 0:
            probability += setup.ml_score * self.params.ml_weight
        
        # Normalize probability to 0-1 range
        setup.probability = min(1.0, probability)
    
    def get_best_setup(self) -> Optional[TradeSetup]:
        """Get the best trade setup based on probability.
        
        Returns:
            Best TradeSetup or None if no valid setups
        """
        valid_setups = [setup for setup in self.trade_setups if setup.is_valid()]
        
        if not valid_setups:
            return None
        
        # Sort by probability (highest first)
        valid_setups.sort(key=lambda x: x.probability, reverse=True)
        
        return valid_setups[0]
    
    def execute_setup(self, setup: TradeSetup) -> bool:
        """Execute a trade setup.
        
        Args:
            setup: TradeSetup to execute
        
        Returns:
            True if execution was successful, False otherwise
        """
        if not setup.is_valid():
            self.logger.warning(f"Cannot execute invalid setup for {setup.symbol}")
            return False
        
        # Execute the trade
        success = self.order_manager.place_order(
            symbol=setup.symbol,
            order_type="MARKET",
            direction=setup.direction,
            volume=self.order_manager.calculate_position_size(setup.symbol, setup.entry_price, setup.stop_loss),
            price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            comment=f"SMC_{setup.timeframe}"
        )
        
        if success:
            # Mark setup as executed
            setup.executed = True
            self.logger.info(f"Executed {setup.symbol} setup with probability {setup.probability:.2f}")
        else:
            self.logger.error(f"Failed to execute {setup.symbol} setup")
        
        return success
        
    def run_on_bar(self, symbol: str, timeframe: int, current_time: datetime, 
                  current_bar: pd.Series, previous_bars: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process a new price bar and generate trading signals.
        
        This method is called by the backtesting engine for each bar in the historical data.
        
        Args:
            symbol: The symbol being processed
            timeframe: The timeframe being processed
            current_time: The timestamp of the current bar
            current_bar: The current price bar data
            previous_bars: All previous bars up to the current one
            
        Returns:
            Optional dictionary with signals and other information, or None if no action
        """
        # For backtesting, implement a simplified strategy
        # This is a basic implementation that can be expanded later
        
        # Need at least 20 bars for analysis
        if len(previous_bars) < 20:
            return None
            
        # Calculate simple moving averages
        fast_ma = previous_bars['close'].rolling(window=10).mean().iloc[-1]
        slow_ma = previous_bars['close'].rolling(window=20).mean().iloc[-1]
        
        # Current price
        current_price = current_bar['close']
        
        # Simple strategy: Buy when fast MA crosses above slow MA
        if fast_ma > slow_ma and current_price > fast_ma:
            # Calculate stop loss and take profit
            stop_loss = current_price * 0.98  # 2% below entry
            take_profit = current_price * 1.04  # 4% above entry
            
            return {
                'signals': [{
                    'type': 'entry',
                    'direction': 1,  # Long
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': 2.0,  # Fixed risk-reward for simplicity
                    'probability': 0.7  # Fixed probability for simplicity
                }]
            }
        
        # Sell when fast MA crosses below slow MA
        elif fast_ma < slow_ma and current_price < fast_ma:
            # Calculate stop loss and take profit
            stop_loss = current_price * 1.02  # 2% above entry
            take_profit = current_price * 0.96  # 4% below entry
            
            return {
                'signals': [{
                    'type': 'entry',
                    'direction': -1,  # Short
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': 2.0,  # Fixed risk-reward for simplicity
                    'probability': 0.7  # Fixed probability for simplicity
                }]
            }
        
        return None