# Connecting to MetaTrader 5 - Implementation Guide

## Overview

This guide provides detailed instructions on how to connect the SMC Liquidity Grab Strategy trading bot to MetaTrader 5 (MT5) for live trading. It covers the necessary setup steps, configuration, and troubleshooting tips.

## Prerequisites

1. **MetaTrader 5 Installation**
   - Download and install MetaTrader 5 from the [official website](https://www.metatrader5.com/en/download)
   - Create or log in to your trading account
   - Ensure your account has API access enabled (contact your broker if needed)

2. **Python MetaTrader 5 Package**
   - Install the required package: `pip install MetaTrader5`
   - Verify installation: `python -c "import MetaTrader5 as mt5; print(mt5.__version__)"`

3. **Trading Account Credentials**
   - Server name
   - Account number
   - Password
   - Investor password (optional, for read-only access)

## Configuration Setup

### 1. Create MT5 Connection Configuration File

Create a new file `mt5_credentials.json` in the `config` directory with the following structure:

```json
{
  "account": {
    "login": YOUR_ACCOUNT_NUMBER,
    "password": "YOUR_PASSWORD",
    "server": "YOUR_BROKER_SERVER",
    "path": "C:/Program Files/MetaTrader 5"
  },
  "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
  "timeframes": ["M15", "H1", "H4", "D1"],
  "max_bars": 5000,
  "retry_attempts": 3,
  "retry_delay": 5
}
```

### 2. Update MT5Config Class

Modify the `mt5_config.py` file to load credentials from the configuration file:

```python
def load_credentials(self):
    """Load MT5 credentials from configuration file."""
    try:
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mt5_credentials.json')
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                self.credentials = json.load(f)
            return True
        else:
            self.logger.error("MT5 credentials file not found")
            return False
    except Exception as e:
        self.logger.error(f"Error loading MT5 credentials: {e}")
        return False
```

## Connection Implementation

### 1. Initialize MT5 Connection

Update the `initialize` method in `mt5_config.py` to use the credentials:

```python
def initialize(self):
    """Initialize connection to MetaTrader 5."""
    try:
        # Load credentials
        if not self.load_credentials():
            return False
        
        # Initialize MT5
        if not mt5.initialize(
            path=self.credentials['account']['path'],
            login=self.credentials['account']['login'],
            password=self.credentials['account']['password'],
            server=self.credentials['account']['server']
        ):
            error = mt5.last_error()
            self.logger.error(f"MT5 initialization failed: {error}")
            return False
        
        # Check connection
        if not mt5.terminal_info():
            self.logger.error("MT5 terminal info not available")
            return False
        
        # Log successful connection
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return False
        
        self.logger.info(f"Connected to MT5: {account_info.server}")
        self.logger.info(f"Account: {account_info.login}, Balance: {account_info.balance}")
        
        return True
    except Exception as e:
        self.logger.error(f"Error initializing MT5: {e}")
        return False
```

### 2. Create Connection Test Script

Create a new file `test_mt5_connection.py` in the project root:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for MetaTrader 5 connection.
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from config.mt5_config import MT5Config
from utils.logger import setup_logging


def test_mt5_connection():
    """Test connection to MetaTrader 5."""
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Testing MT5 connection")
    
    # Initialize MT5 connection
    mt5_config = MT5Config()
    if not mt5_config.initialize():
        logger.error("Failed to initialize MT5")
        return False
    
    # Get account information
    account_info = mt5_config.get_account_info()
    logger.info(f"Account: {account_info['login']}")
    logger.info(f"Name: {account_info['name']}")
    logger.info(f"Server: {account_info['server']}")
    logger.info(f"Balance: {account_info['balance']}")
    logger.info(f"Equity: {account_info['equity']}")
    
    # Get available symbols
    symbols = mt5_config.get_symbols()
    logger.info(f"Available symbols: {', '.join(symbols[:10])}...")
    
    # Shutdown MT5
    mt5_config.shutdown()
    logger.info("MT5 connection test completed successfully")
    return True


if __name__ == "__main__":
    success = test_mt5_connection()
    sys.exit(0 if success else 1)
```

## Live Trading Implementation

### 1. Update Main Trading Loop

Modify the `main.py` file to implement a complete trading loop:

```python
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
    
    # Create ML predictor if enabled
    ml_predictor = None
    if params.ml_params.enabled:
        ml_predictor = MLPredictor(params.ml_params)
    
    # Define trading parameters
    symbols = params.backtest_params.symbols
    timeframes = [getattr(mt5, f"TIMEFRAME_{tf}") for tf in params.backtest_params.timeframes]
    
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
                        
                        # Apply ML prediction if enabled
                        if setup is not None and ml_predictor is not None:
                            prediction = ml_predictor.predict(symbol, timeframe, df)
                            if prediction['success']:
                                # Adjust setup probability based on ML prediction
                                ml_score = prediction['probability'] if prediction['prediction'] == (1 if setup.direction == 'long' else 0) else 1 - prediction['probability']
                                setup.probability = smc_strategy._calculate_setup_probability(setup, ml_score)
                        
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
```

### 2. Update Main Function

Update the main function to handle command-line arguments:

```python
def main():
    """Main function to run the trading bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SMC Liquidity Grab Strategy Trading Bot')
    parser.add_argument('--config', type=str, default='config/default_params.json',
                        help='Path to configuration file')
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize strategy parameters')
    
    args = parser.parse_args()
    
    if args.backtest:
        from example import run_backtest_example
        run_backtest_example(args.config)
    elif args.optimize:
        from example import run_optimization_example
        run_optimization_example(args.config)
    else:
        run_trading_bot(args.config)


if __name__ == "__main__":
    main()
```

## Troubleshooting MT5 Connection

### Common Issues and Solutions

1. **Connection Errors**
   - Verify your internet connection
   - Check if MT5 is running and logged in
   - Ensure your account has API access enabled
   - Verify server name, login, and password

2. **Symbol Not Found**
   - Check if the symbol is available in your MT5 terminal
   - Verify symbol naming (some brokers use different naming conventions)
   - Use `mt5.symbols_get()` to get a list of available symbols

3. **Order Execution Failures**
   - Check if you have sufficient margin
   - Verify trading is allowed for the symbol
   - Check if the market is open
   - Ensure position size is within limits

4. **Data Retrieval Issues**
   - Check if historical data is available for the symbol and timeframe
   - Reduce the number of bars requested
   - Increase retry attempts for data fetching

### Debugging Tips

1. Enable detailed logging:
   ```python
   setup_logging(log_level=logging.DEBUG)
   ```

2. Check MT5 terminal logs for additional error information

3. Use the `test_mt5_connection.py` script to verify connectivity

4. Monitor system resources (memory, CPU) during execution

## Best Practices for Live Trading

1. **Start with Demo Account**
   - Always test on a demo account before using real funds
   - Verify all functionality works as expected

2. **Risk Management**
   - Set appropriate risk limits in `risk_params`
   - Monitor drawdown and adjust parameters if needed
   - Implement circuit breakers for unexpected market conditions

3. **Monitoring**
   - Set up alerts for critical errors
   - Regularly check logs for warnings or issues
   - Monitor trade execution and performance

4. **Backup and Recovery**
   - Regularly backup configuration files
   - Implement error recovery mechanisms
   - Have a plan for handling connection losses

## Conclusion

By following this guide, you should now have a fully functional trading bot connected to MetaTrader 5. The implementation includes proper initialization, error handling, and a complete trading loop that analyzes the market, identifies trade setups, and executes trades based on the SMC Liquidity Grab Strategy.

For further assistance or troubleshooting, refer to the [MetaTrader 5 Python API documentation](https://www.mql5.com/en/docs/python_metatrader5) or contact your broker's support team.