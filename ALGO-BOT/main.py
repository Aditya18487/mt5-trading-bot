#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the MT5 Algorithmic Trading Bot.
This script initializes the trading environment, connects to MT5,
and orchestrates the trading strategy execution.

Usage:
    python main.py                      # Run in live trading mode
    python main.py --backtest           # Run in backtest mode
    python main.py --optimize           # Run parameter optimization
    python main.py --config config.json # Use custom config file
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config.mt5_config import MT5Config
from config.strategy_params import StrategyParameters

# Import core modules
from data.market_data import MarketData
from strategies.smc_strategy import SMCStrategy
from execution.order_manager import OrderManager
from utils.logger import setup_logging, BotLogger
from utils.risk_manager import RiskManager
#from backtesting.backtest_engine import BacktestEngine

# ML module has been removed
ML_AVAILABLE = False

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)


def run_trading_bot(config_file='config/default_params.json'):
    """Run the trading bot with the specified configuration."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting SMC Liquidity Grab Strategy trading bot")
    
    # Load parameters
    params = StrategyParameters()
    params.load_from_file(config_file)
    
    # Initialize MT5 connection
    mt5_config = MT5Config()
    if not mt5_config.initialize():
        logger.error("Failed to initialize MT5")
        return
    
    # Create components
    market_data = MarketData(mt5_config)
    order_manager = OrderManager(mt5_config)
    risk_manager = RiskManager(mt5_config, params.risk_params)
    
    # Create strategy instance
    smc_strategy = SMCStrategy(
        params.smc_params, 
        market_data, 
        order_manager, 
        risk_manager
    )
    
    # ML predictor has been removed
    ml_predictor = None
    
    # Define trading parameters
    symbols = params.backtest_params.symbols
    timeframes = [mt5_config.get_timeframe_by_name(tf) for tf in params.backtest_params.timeframes]
    
    # Trading loop
    try:
        logger.info("Starting trading loop")
        running = True
        
        while running:
            try:
                # Check if market is open
                if not market_data.is_market_open():
                    logger.info("Market is closed. Waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Process each symbol and timeframe
                for symbol in symbols:
                    for timeframe in timeframes:
                        # Update market data
                        df = market_data.get_latest_bars(symbol, timeframe, 200)
                        
                        if df is None or len(df) < 100:
                            logger.warning(f"Insufficient data for {symbol} on {timeframe}")
                            continue
                        
                        # Analyze market and find setups
                        smc_strategy.analyze_market(df, symbol, timeframe)
                        
                        # Get best setup
                        setup = smc_strategy.get_best_setup()
                        
                        # ML prediction has been removed
                        
                        # Execute setup if valid
                        if setup is not None and setup.probability >= params.smc_params.min_setup_score / 100:
                            # Validate with risk manager
                            if risk_manager.validate_trade(setup):
                                logger.info(f"Executing setup for {symbol} on {timeframe}")
                                result = smc_strategy.execute_setup(setup)
                                
                                if result['success']:
                                    logger.info(f"Trade executed: {result['order_id']}")
                                else:
                                    logger.error(f"Trade execution failed: {result['error']}")
                            else:
                                logger.info(f"Setup rejected by risk manager for {symbol}")
                
                # Manage open positions
                open_positions = order_manager.get_open_positions()
                for position in open_positions:
                    # Check for trailing stop adjustment
                    if params.risk_params.trailing_stop_activation > 0:
                        order_manager.update_trailing_stop(position)
                
                # Wait for next check
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
                running = False
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)  # Wait before retrying
    
    finally:
        # Shutdown MT5
        mt5_config.shutdown()
        logger.info("Trading bot stopped")


def main():
    """Main function to run the trading bot."""
    # Setup logging
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='SMC Liquidity Grab Strategy Trading Bot')
    parser.add_argument('--config', type=str, default='config/default_params.json',
                        help='Path to configuration file')
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize strategy parameters')
    
    args = parser.parse_args()
    
    try:
        if args.backtest or args.optimize:
            logger.info("Backtesting and optimization functionality has been removed.")
            logger.info("Please use the main trading bot functionality.")
        else:
            run_trading_bot(args.config)
            
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        # Clean up resources
        logger.info("MT5 connection closed. Trading bot stopped.")


if __name__ == "__main__":
    main()