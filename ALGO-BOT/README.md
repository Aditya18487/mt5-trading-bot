# Advanced Algorithmic Trading Bot for MT5

A hedge fund-level algorithmic trading bot for MetaTrader 5 (MT5) that implements Smart Money Concepts (SMC) and advanced price action strategies. This bot identifies high-probability trading setups by detecting liquidity zones, market structure breaks, and other technical patterns, and can execute trades automatically through MetaTrader 5.

## Features

- **Liquidity Zone Detection**
  - Swing highs/lows identification
  - Order blocks with imbalance detection
  - Fair Value Gaps (FVG)
  - Consolidation zones (liquidity pools)

- **Candlestick Pattern Recognition**
  - Bullish/Bearish Engulfing
  - Pin Bars
  - Inside Bars
  - Doji with context

- **Price Action & Market Structure Analysis**
  - Break of Structure (BOS), Change of Character (CHoCH)
  - Higher highs/lows, lower highs/lows
  - Smart entry after liquidity sweep + structure shift

- **Chart Pattern Detection**
  - Double tops/bottoms
  - Triangles, flags
  - Head & shoulders

- **Advanced Trade Execution**
  - Smart entry post-liquidity sweep
  - Strategic stop-loss placement
  - Dynamic take-profit targeting
  - Position sizing based on account risk %
  - Trailing stop-loss and breakeven features
  - Silent execution using limit/staggered orders

- **Risk Management**
  - Max loss per trade and per day
  - Daily drawdown protection
  - Comprehensive trade logging

- **Backtesting Support**
  - Visualization of candles, liquidity zones, entries/exits
  - Performance metrics and statistics

- **Optional ML Module**
  - Trade setup scoring using LightGBM/Scikit-learn

## Project Structure

```
ALGO-BOT/
├── backtesting/        # Backtesting engine and results
│   └── backtest_engine.py
├── config/             # Configuration parameters
│   ├── default_params.json
│   ├── mt5_config.py
│   ├── strategy_params.py
├── data/               # OHLC and volume data
│   └── market_data.py
├── docs/               # Documentation
│   └── mt5_connection_guide.md
├── execution/          # Order handling and execution
│   └── order_manager.py
├── logs/               # Trade and system logs
├── ml/                 # Optional machine learning module
│   └── predictor.py
├── strategy/           # Trading strategy logic modules
│   └── smc_strategy.py
├── utils/              # Helper functions and utilities
│   ├── logger.py
│   ├── risk_manager.py
│   ├── trading_utils.py
│   └── visualization.py
├── example.py          # Example usage
├── main.py             # Main entry point
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
└── test_mt5_connection.py # MT5 connection test
```

## Getting Started

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your MT5 connection:
   - Create `config/mt5_credentials.json` with your account details (see template in docs)
   - Run the connection test: `python test_mt5_connection.py`

3. Adjust strategy parameters in `config/default_params.json`

4. Run the bot:
   - For backtesting: `python main.py --backtest --config config/default_params.json`
   - For optimization: `python main.py --optimize --config config/default_params.json`
   - For live trading: `python main.py --config config/default_params.json`

For detailed instructions on connecting to MT5, refer to the `docs/mt5_connection_guide.md` file.

## Requirements

- Python 3.8+
- MetaTrader5 Python API
- NumPy, Pandas
- Matplotlib, Plotly, Seaborn (for visualization)
- scikit-learn, XGBoost, LightGBM (for ML module)
- TA-Lib (for technical indicators)
- Windows operating system (required for MT5)

See `requirements.txt` for the complete list of dependencies.