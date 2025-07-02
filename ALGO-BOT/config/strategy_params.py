#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy parameters configuration.

This module defines all adjustable parameters for the trading strategies,
risk management, and backtesting settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import MetaTrader5 as mt5


@dataclass
class SMCParams:
    """Parameters for Smart Money Concepts (SMC) strategy."""
    # Liquidity zone detection parameters
    lookback_period: int = 20  # Number of bars to look back for swing points
    swing_strength: float = 0.5  # Minimum strength for swing high/low (0-1)
    liquidity_zone_padding: float = 0.0002  # Padding for liquidity zones in price units
    
    # Order block detection parameters
    ob_candle_count: int = 3  # Number of candles to consider for order blocks
    ob_min_body_size: float = 0.0005  # Minimum body size for order block candles
    ob_imbalance_threshold: float = 0.7  # Threshold for imbalance detection (0-1)
    
    # Fair Value Gap (FVG) parameters
    fvg_min_gap_size: float = 0.0003  # Minimum gap size for FVG detection
    fvg_max_candles: int = 50  # Maximum candles to keep FVG valid
    
    # Consolidation zone parameters
    consolidation_range_threshold: float = 0.0015  # Maximum range for consolidation
    consolidation_min_candles: int = 5  # Minimum candles for consolidation
    
    # Candlestick pattern parameters
    engulfing_body_ratio: float = 1.5  # Ratio for engulfing pattern body size
    pin_bar_nose_ratio: float = 0.6  # Ratio of nose to total candle for pin bar
    doji_body_threshold: float = 0.0001  # Maximum body size for doji
    
    # Market structure parameters
    bos_confirmation_candles: int = 2  # Candles to confirm break of structure
    structure_swing_lookback: int = 10  # Lookback for structure analysis
    
    # Chart pattern parameters
    pattern_recognition_enabled: bool = True  # Enable chart pattern recognition
    pattern_min_candles: int = 5  # Minimum candles for pattern formation
    pattern_max_candles: int = 50  # Maximum candles for pattern formation
    
    # Entry parameters
    require_liquidity_sweep: bool = True  # Require liquidity sweep before entry
    require_structure_shift: bool = True  # Require structure shift before entry
    entry_confirmation_candles: int = 1  # Candles to wait after setup for entry
    
    # Timeframes to analyze
    timeframes: List[int] = field(default_factory=lambda: [
        mt5.TIMEFRAME_M5,
        mt5.TIMEFRAME_M15,
        mt5.TIMEFRAME_H1,
        mt5.TIMEFRAME_H4,
        mt5.TIMEFRAME_D1
    ])


@dataclass
class RiskParams:
    """Parameters for risk management."""
    # Position sizing
    risk_per_trade: float = 0.01  # Risk per trade as fraction of account (1% default)
    max_risk_per_day: float = 0.05  # Maximum risk per day (5% default)
    max_trades_per_day: int = 5  # Maximum number of trades per day
    max_positions: int = 5  # Maximum number of positions per symbol
    max_risk_per_symbol: float = 0.02  # Maximum risk per symbol as fraction of account (2% default)
    
    # Stop loss and take profit
    default_risk_reward: float = 2.0  # Default risk-reward ratio
    min_stop_distance_pips: int = 10  # Minimum stop distance in pips
    max_stop_distance_pips: int = 100  # Maximum stop distance in pips
    
    # Trailing stop and breakeven
    use_trailing_stop: bool = True  # Use trailing stop
    trailing_stop_activation: float = 0.5  # Activate trailing stop after this fraction of TP reached
    trailing_stop_distance: float = 0.3  # Trailing stop distance as fraction of ATR
    
    # Breakeven settings
    use_breakeven: bool = True  # Move stop loss to breakeven
    breakeven_activation: float = 0.5  # Move to breakeven after this fraction of TP reached
    breakeven_padding_pips: int = 2  # Pips of profit when moving to breakeven
    
    # Drawdown protection
    max_daily_drawdown: float = 0.05  # Maximum daily drawdown (5% default)
    pause_after_consecutive_losses: int = 3  # Pause after this many consecutive losses
    
    # Execution settings
    use_limit_orders: bool = True  # Use limit orders instead of market orders
    use_staggered_entries: bool = False  # Use staggered entries
    staggered_entry_levels: int = 3  # Number of staggered entry levels
    staggered_entry_spacing_pips: int = 5  # Spacing between staggered entries in pips
    slippage_points: int = 10  # Maximum allowed slippage in points


@dataclass
class BacktestParams:
    """Parameters for backtesting."""
    start_date: str = "2023-01-01"  # Start date for backtesting
    end_date: str = "2023-12-31"  # End date for backtesting
    initial_deposit: float = 10000.0  # Initial deposit for backtesting
    initial_balance: float = 10000.0  # Initial balance for backtesting (used by BacktestEngine)
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    timeframes: List[int] = field(default_factory=lambda: [mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4])
    commission_per_lot: float = 7.0  # Commission per standard lot
    spread_adjustment: bool = True  # Adjust for historical spread
    slippage_pips: int = 1  # Slippage in pips for backtesting
    visualization_enabled: bool = True  # Enable visualization of backtest results


@dataclass
class MLParams:
    """Parameters for machine learning module."""
    enabled: bool = False  # Enable ML scoring
    model_type: str = "lightgbm"  # Model type (lightgbm, sklearn)
    model_path: str = "ml/models/setup_scorer.pkl"  # Path to trained model
    feature_importance_threshold: float = 0.01  # Minimum feature importance
    min_score_for_trade: float = 0.6  # Minimum score to consider trade valid
    retrain_interval_days: int = 30  # Retrain model every X days


class StrategyParameters:
    """Container for all strategy parameters."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize strategy parameters.
        
        Args:
            config_file: Path to JSON configuration file (optional)
        """
        # Initialize default parameters
        self.smc_params = SMCParams()
        self.risk_params = RiskParams()
        self.backtest_params = BacktestParams()
        self.ml_params = MLParams()
        
        # General settings
        self.update_interval = 5  # Seconds between market data updates
        self.use_ml_scoring = False  # Use ML to score trade setups
        self.ml_model_path = "ml/models/setup_scorer.pkl"  # Path to ML model
        
        # Load from config file if provided
        if config_file:
            self._load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load parameters from JSON configuration file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self._load_from_file(config_file)
    
    def _load_from_file(self, config_file: str) -> None:
        """Load parameters from JSON configuration file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        import json
        import logging
        logger = logging.getLogger('main_logger')
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update parameters from config
            if 'smc_params' in config:
                for key, value in config['smc_params'].items():
                    if hasattr(self.smc_params, key):
                        setattr(self.smc_params, key, value)
            
            if 'risk_params' in config:
                for key, value in config['risk_params'].items():
                    if hasattr(self.risk_params, key):
                        setattr(self.risk_params, key, value)
            
            if 'backtest_params' in config:
                for key, value in config['backtest_params'].items():
                    if hasattr(self.backtest_params, key):
                        setattr(self.backtest_params, key, value)
            
            if 'ml_params' in config:
                for key, value in config['ml_params'].items():
                    if hasattr(self.ml_params, key):
                        setattr(self.ml_params, key, value)
            
            # Update general settings
            if 'update_interval' in config:
                self.update_interval = config['update_interval']
            
            if 'use_ml_scoring' in config:
                self.use_ml_scoring = config['use_ml_scoring']
            
            if 'ml_model_path' in config:
                self.ml_model_path = config['ml_model_path']
            
            logger.info(f"Strategy parameters loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading strategy parameters: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """Save parameters to JSON configuration file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        import json
        import logging
        from dataclasses import asdict
        logger = logging.getLogger('main_logger')
        
        try:
            # Convert dataclasses to dictionaries
            config = {
                'smc_params': asdict(self.smc_params),
                'risk_params': asdict(self.risk_params),
                'backtest_params': asdict(self.backtest_params),
                'ml_params': asdict(self.ml_params),
                'update_interval': self.update_interval,
                'use_ml_scoring': self.use_ml_scoring,
                'ml_model_path': self.ml_model_path
            }
            
            # Convert non-serializable objects (like MT5 timeframe constants)
            for section in config.values():
                if isinstance(section, dict):
                    for key, value in section.items():
                        if isinstance(value, list) and value and not isinstance(value[0], (str, int, float, bool, type(None))):
                            section[key] = [str(item) for item in value]
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Strategy parameters saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving strategy parameters: {e}")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create default parameters
    params = StrategyParameters()
    
    # Save to file
    params.save_to_file("config/strategy_params.json")
    
    # Load from file
    loaded_params = StrategyParameters("config/strategy_params.json")