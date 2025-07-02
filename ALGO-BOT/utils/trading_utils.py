#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading utilities module for the algorithmic trading bot.

This module provides:
- Technical indicator calculations
- Fibonacci levels and pivot points
- Price action pattern detection
- Trading session identification
- Time and date utilities for trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time, timedelta
import pytz


class TechnicalIndicators:
    """Technical indicator calculations."""
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average.
        
        Args:
            data: Price data array
            period: SMA period
        
        Returns:
            SMA values
        """
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Price data array
            period: EMA period
        
        Returns:
            EMA values
        """
        alpha = 2 / (period + 1)
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]  # Initialize with first value
        
        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        
        return ema_values
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index.
        
        Args:
            data: Price data array
            period: RSI period
        
        Returns:
            RSI values
        """
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        
        # Initialize first values
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Calculate smoothed averages
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        # Calculate RS and RSI
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi_values = 100 - (100 / (1 + rs))
        
        return rsi_values
    
    @staticmethod
    def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data array
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        fast_ema = TechnicalIndicators.ema(data, fast_period)
        slow_ema = TechnicalIndicators.ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands.
        
        Args:
            data: Price data array
            period: Moving average period
            std_dev: Standard deviation multiplier
        
        Returns:
            Tuple of (upper band, middle band, lower band)
        """
        middle_band = TechnicalIndicators.sma(data, period)
        rolling_std = np.array([np.std(data[max(0, i-period+1):i+1]) for i in range(len(data))])
        
        upper_band = middle_band + (rolling_std[-len(middle_band):] * std_dev)
        lower_band = middle_band - (rolling_std[-len(middle_band):] * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ATR period
        
        Returns:
            ATR values
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # First value has no previous close
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr_values = np.zeros_like(close)
        atr_values[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(close)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
        
        return atr_values
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            k_period: %K period
            d_period: %D period
        
        Returns:
            Tuple of (%K, %D)
        """
        k_values = np.zeros_like(close)
        
        for i in range(k_period - 1, len(close)):
            window_high = np.max(high[i-k_period+1:i+1])
            window_low = np.min(low[i-k_period+1:i+1])
            k_values[i] = 100 * (close[i] - window_low) / (window_high - window_low) if window_high != window_low else 50
        
        d_values = TechnicalIndicators.sma(k_values[k_period-1:], d_period)
        
        return k_values, np.append(np.zeros(len(k_values) - len(d_values)), d_values)
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Average Directional Index.
        
        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ADX period
        
        Returns:
            Tuple of (ADX, +DI, -DI, DX)
        """
        # True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # +DM and -DM
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        smoothed_tr = np.zeros_like(close)
        smoothed_plus_dm = np.zeros_like(close)
        smoothed_minus_dm = np.zeros_like(close)
        
        # Initialize
        smoothed_tr[period-1] = np.sum(tr[:period])
        smoothed_plus_dm[period-1] = np.sum(plus_dm[:period])
        smoothed_minus_dm[period-1] = np.sum(minus_dm[:period])
        
        # Calculate smoothed values
        for i in range(period, len(close)):
            smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx_values = np.zeros_like(close)
        adx_values[2*period-1] = np.mean(dx[period:2*period])
        
        for i in range(2*period, len(close)):
            adx_values[i] = (adx_values[i-1] * (period - 1) + dx[i]) / period
        
        return adx_values, plus_di, minus_di, dx


class FibonacciLevels:
    """Fibonacci retracement and extension levels."""
    
    @staticmethod
    def retracement_levels(high_price: float, low_price: float, is_uptrend: bool = True) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels.
        
        Args:
            high_price: High price point
            low_price: Low price point
            is_uptrend: True if trend is up, False if down
        
        Returns:
            Dictionary of Fibonacci levels
        """
        diff = high_price - low_price
        
        if is_uptrend:
            # Uptrend: retracements are below high price
            return {
                '0.0': high_price,
                '0.236': high_price - 0.236 * diff,
                '0.382': high_price - 0.382 * diff,
                '0.5': high_price - 0.5 * diff,
                '0.618': high_price - 0.618 * diff,
                '0.786': high_price - 0.786 * diff,
                '1.0': low_price
            }
        else:
            # Downtrend: retracements are above low price
            return {
                '0.0': low_price,
                '0.236': low_price + 0.236 * diff,
                '0.382': low_price + 0.382 * diff,
                '0.5': low_price + 0.5 * diff,
                '0.618': low_price + 0.618 * diff,
                '0.786': low_price + 0.786 * diff,
                '1.0': high_price
            }
    
    @staticmethod
    def extension_levels(start_price: float, end_price: float, retracement_price: float) -> Dict[str, float]:
        """Calculate Fibonacci extension levels.
        
        Args:
            start_price: Starting price point
            end_price: Ending price point (swing high/low)
            retracement_price: Price after retracement
        
        Returns:
            Dictionary of Fibonacci extension levels
        """
        is_uptrend = end_price > start_price
        diff = abs(end_price - start_price)
        
        if is_uptrend:
            # Uptrend: extensions are above end price
            return {
                '0.0': retracement_price,
                '0.618': retracement_price + 0.618 * diff,
                '1.0': retracement_price + 1.0 * diff,
                '1.618': retracement_price + 1.618 * diff,
                '2.0': retracement_price + 2.0 * diff,
                '2.618': retracement_price + 2.618 * diff,
                '3.618': retracement_price + 3.618 * diff
            }
        else:
            # Downtrend: extensions are below end price
            return {
                '0.0': retracement_price,
                '0.618': retracement_price - 0.618 * diff,
                '1.0': retracement_price - 1.0 * diff,
                '1.618': retracement_price - 1.618 * diff,
                '2.0': retracement_price - 2.0 * diff,
                '2.618': retracement_price - 2.618 * diff,
                '3.618': retracement_price - 3.618 * diff
            }


class PivotPoints:
    """Pivot point calculations."""
    
    @staticmethod
    def standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate standard pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
        
        Returns:
            Dictionary of pivot points
        """
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'R3': r3,
            'R2': r2,
            'R1': r1,
            'P': pivot,
            'S1': s1,
            'S2': s2,
            'S3': s3
        }
    
    @staticmethod
    def camarilla_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Camarilla pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
        
        Returns:
            Dictionary of Camarilla pivot points
        """
        diff = high - low
        
        r4 = close + diff * 1.1/2
        r3 = close + diff * 1.1/4
        r2 = close + diff * 1.1/6
        r1 = close + diff * 1.1/12
        
        s1 = close - diff * 1.1/12
        s2 = close - diff * 1.1/6
        s3 = close - diff * 1.1/4
        s4 = close - diff * 1.1/2
        
        return {
            'R4': r4,
            'R3': r3,
            'R2': r2,
            'R1': r1,
            'PP': close,
            'S1': s1,
            'S2': s2,
            'S3': s3,
            'S4': s4
        }
    
    @staticmethod
    def woodie_pivot_points(high: float, low: float, close: float, open_price: float) -> Dict[str, float]:
        """Calculate Woodie pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            open_price: Current period open
        
        Returns:
            Dictionary of Woodie pivot points
        """
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + high - low
        s2 = pivot - high + low
        
        return {
            'R2': r2,
            'R1': r1,
            'PP': pivot,
            'S1': s1,
            'S2': s2
        }


class PricePatterns:
    """Price action pattern detection."""
    
    @staticmethod
    def is_engulfing(open_prices: np.ndarray, close_prices: np.ndarray, index: int) -> Tuple[bool, str]:
        """Check if candle at index is an engulfing pattern.
        
        Args:
            open_prices: Array of open prices
            close_prices: Array of close prices
            index: Index to check
        
        Returns:
            Tuple of (is_engulfing, pattern_type)
        """
        if index < 1 or index >= len(open_prices):
            return False, ""
        
        curr_open = open_prices[index]
        curr_close = close_prices[index]
        prev_open = open_prices[index-1]
        prev_close = close_prices[index-1]
        
        curr_body_size = abs(curr_close - curr_open)
        prev_body_size = abs(prev_close - prev_open)
        
        # Bullish engulfing
        if (prev_close < prev_open and  # Previous candle is bearish
            curr_close > curr_open and  # Current candle is bullish
            curr_open <= prev_close and  # Current open is below or equal to previous close
            curr_close >= prev_open and  # Current close is above or equal to previous open
            curr_body_size > prev_body_size):  # Current body is larger
            return True, "bullish"
        
        # Bearish engulfing
        elif (prev_close > prev_open and  # Previous candle is bullish
              curr_close < curr_open and  # Current candle is bearish
              curr_open >= prev_close and  # Current open is above or equal to previous close
              curr_close <= prev_open and  # Current close is below or equal to previous open
              curr_body_size > prev_body_size):  # Current body is larger
            return True, "bearish"
        
        return False, ""
    
    @staticmethod
    def is_doji(open_price: float, high_price: float, low_price: float, close_price: float, doji_threshold: float = 0.1) -> bool:
        """Check if candle is a doji.
        
        Args:
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            doji_threshold: Maximum body to range ratio for doji
        
        Returns:
            True if doji, False otherwise
        """
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0:
            return False
        
        body_to_range_ratio = body_size / range_size
        
        return body_to_range_ratio <= doji_threshold
    
    @staticmethod
    def is_hammer(open_price: float, high_price: float, low_price: float, close_price: float) -> bool:
        """Check if candle is a hammer.
        
        Args:
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
        
        Returns:
            True if hammer, False otherwise
        """
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0 or body_size == 0:
            return False
        
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Hammer criteria:
        # 1. Lower shadow is at least 2x the body size
        # 2. Upper shadow is small (less than 10% of range)
        # 3. Body is in the upper 1/3 of the range
        return (lower_shadow >= 2 * body_size and
                upper_shadow <= 0.1 * range_size and
                body_bottom >= (low_price + 0.67 * range_size))
    
    @staticmethod
    def is_shooting_star(open_price: float, high_price: float, low_price: float, close_price: float) -> bool:
        """Check if candle is a shooting star.
        
        Args:
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
        
        Returns:
            True if shooting star, False otherwise
        """
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0 or body_size == 0:
            return False
        
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Shooting star criteria:
        # 1. Upper shadow is at least 2x the body size
        # 2. Lower shadow is small (less than 10% of range)
        # 3. Body is in the lower 1/3 of the range
        return (upper_shadow >= 2 * body_size and
                lower_shadow <= 0.1 * range_size and
                body_top <= (low_price + 0.33 * range_size))
    
    @staticmethod
    def is_inside_bar(high_prices: np.ndarray, low_prices: np.ndarray, index: int) -> bool:
        """Check if candle at index is an inside bar.
        
        Args:
            high_prices: Array of high prices
            low_prices: Array of low prices
            index: Index to check
        
        Returns:
            True if inside bar, False otherwise
        """
        if index < 1 or index >= len(high_prices):
            return False
        
        curr_high = high_prices[index]
        curr_low = low_prices[index]
        prev_high = high_prices[index-1]
        prev_low = low_prices[index-1]
        
        return curr_high <= prev_high and curr_low >= prev_low
    
    @staticmethod
    def is_outside_bar(high_prices: np.ndarray, low_prices: np.ndarray, index: int) -> bool:
        """Check if candle at index is an outside bar.
        
        Args:
            high_prices: Array of high prices
            low_prices: Array of low prices
            index: Index to check
        
        Returns:
            True if outside bar, False otherwise
        """
        if index < 1 or index >= len(high_prices):
            return False
        
        curr_high = high_prices[index]
        curr_low = low_prices[index]
        prev_high = high_prices[index-1]
        prev_low = low_prices[index-1]
        
        return curr_high > prev_high and curr_low < prev_low
    
    @staticmethod
    def is_pinbar(open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, 
                  close_prices: np.ndarray, index: int, nose_ratio: float = 0.33) -> Tuple[bool, str]:
        """Check if candle at index is a pin bar (price rejection).
        
        Args:
            open_prices: Array of open prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            close_prices: Array of close prices
            index: Index to check
            nose_ratio: Minimum ratio of nose to body for pin bar
        
        Returns:
            Tuple of (is_pinbar, direction)
        """
        if index >= len(open_prices):
            return False, ""
        
        open_price = open_prices[index]
        high_price = high_prices[index]
        low_price = low_prices[index]
        close_price = close_prices[index]
        
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0 or body_size == 0:
            return False, ""
        
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Bullish pin bar (rejection of lower prices)
        if (lower_shadow >= nose_ratio * range_size and
            lower_shadow >= 2 * body_size and
            upper_shadow <= 0.2 * range_size):
            return True, "bullish"
        
        # Bearish pin bar (rejection of higher prices)
        elif (upper_shadow >= nose_ratio * range_size and
              upper_shadow >= 2 * body_size and
              lower_shadow <= 0.2 * range_size):
            return True, "bearish"
        
        return False, ""


class TradingSession:
    """Trading session identification and time utilities."""
    
    # Define major trading sessions in UTC
    SYDNEY_SESSION = (time(21, 0), time(6, 0))  # 21:00 - 06:00 UTC
    TOKYO_SESSION = (time(0, 0), time(9, 0))    # 00:00 - 09:00 UTC
    LONDON_SESSION = (time(8, 0), time(16, 0))  # 08:00 - 16:00 UTC
    NEW_YORK_SESSION = (time(13, 0), time(22, 0))  # 13:00 - 22:00 UTC
    
    @staticmethod
    def get_current_session(dt: datetime = None) -> List[str]:
        """Get current active trading sessions.
        
        Args:
            dt: Datetime to check (default: current UTC time)
        
        Returns:
            List of active sessions
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        
        current_time = dt.time()
        active_sessions = []
        
        # Check Sydney session (crosses midnight)
        if TradingSession._is_in_session(current_time, TradingSession.SYDNEY_SESSION):
            active_sessions.append("Sydney")
        
        # Check Tokyo session
        if TradingSession._is_in_session(current_time, TradingSession.TOKYO_SESSION):
            active_sessions.append("Tokyo")
        
        # Check London session
        if TradingSession._is_in_session(current_time, TradingSession.LONDON_SESSION):
            active_sessions.append("London")
        
        # Check New York session
        if TradingSession._is_in_session(current_time, TradingSession.NEW_YORK_SESSION):
            active_sessions.append("New York")
        
        return active_sessions
    
    @staticmethod
    def _is_in_session(current_time: time, session_times: Tuple[time, time]) -> bool:
        """Check if current time is within session.
        
        Args:
            current_time: Time to check
            session_times: Tuple of (start_time, end_time)
        
        Returns:
            True if in session, False otherwise
        """
        start_time, end_time = session_times
        
        # Session crosses midnight
        if start_time > end_time:
            return current_time >= start_time or current_time <= end_time
        # Normal session
        else:
            return start_time <= current_time <= end_time
    
    @staticmethod
    def is_session_overlap(dt: datetime = None) -> bool:
        """Check if current time is in a session overlap period.
        
        Args:
            dt: Datetime to check (default: current UTC time)
        
        Returns:
            True if in session overlap, False otherwise
        """
        active_sessions = TradingSession.get_current_session(dt)
        return len(active_sessions) > 1
    
    @staticmethod
    def is_high_volatility_time(dt: datetime = None) -> bool:
        """Check if current time is typically a high volatility period.
        
        Args:
            dt: Datetime to check (default: current UTC time)
        
        Returns:
            True if high volatility time, False otherwise
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        
        current_time = dt.time()
        
        # London open (8:00-9:00 UTC)
        london_open = time(8, 0) <= current_time <= time(9, 0)
        
        # New York open (13:00-14:00 UTC)
        ny_open = time(13, 0) <= current_time <= time(14, 0)
        
        # London/New York overlap (13:00-16:00 UTC)
        london_ny_overlap = time(13, 0) <= current_time <= time(16, 0)
        
        # US economic news (typically 12:30-16:00 UTC)
        us_news_time = time(12, 30) <= current_time <= time(16, 0)
        
        return london_open or ny_open or london_ny_overlap or us_news_time
    
    @staticmethod
    def is_weekend(dt: datetime = None) -> bool:
        """Check if date is a weekend.
        
        Args:
            dt: Datetime to check (default: current UTC time)
        
        Returns:
            True if weekend, False otherwise
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        
        # 5 = Saturday, 6 = Sunday
        return dt.weekday() >= 5
    
    @staticmethod
    def time_to_next_session(target_session: str, dt: datetime = None) -> timedelta:
        """Calculate time until next session starts.
        
        Args:
            target_session: Session name ("Sydney", "Tokyo", "London", "New York")
            dt: Datetime to check (default: current UTC time)
        
        Returns:
            Timedelta until next session
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        
        current_time = dt.time()
        
        if target_session == "Sydney":
            session_start = TradingSession.SYDNEY_SESSION[0]
        elif target_session == "Tokyo":
            session_start = TradingSession.TOKYO_SESSION[0]
        elif target_session == "London":
            session_start = TradingSession.LONDON_SESSION[0]
        elif target_session == "New York":
            session_start = TradingSession.NEW_YORK_SESSION[0]
        else:
            raise ValueError(f"Unknown session: {target_session}")
        
        # Convert times to datetime for comparison
        now = datetime.combine(dt.date(), current_time)
        session_start_dt = datetime.combine(dt.date(), session_start)
        
        # If session start is earlier today, move to tomorrow
        if session_start_dt <= now:
            session_start_dt += timedelta(days=1)
        
        return session_start_dt - now