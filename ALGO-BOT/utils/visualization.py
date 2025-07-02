#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for the algorithmic trading bot.

This module provides:
- Chart generation for price data with indicators
- Trade visualization on charts
- Performance metrics visualization
- Backtest result plotting
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import os
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")


class ChartGenerator:
    """Generate charts for price data and trades."""
    
    def __init__(self, save_dir: str = None):
        """Initialize ChartGenerator.
        
        Args:
            save_dir: Directory to save charts (default: logs/charts)
        """
        if save_dir is None:
            self.save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'charts')
        else:
            self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_price_chart(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                         indicators: Dict[str, Dict] = None, trades: List[Dict] = None,
                         liquidity_zones: List[Dict] = None, save: bool = True) -> plt.Figure:
        """Plot price chart with indicators and trades.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            indicators: Dictionary of indicators to plot
            trades: List of trade dictionaries
            liquidity_zones: List of liquidity zone dictionaries
            save: Whether to save the chart
        
        Returns:
            Matplotlib figure
        """
        # Create figure and axes
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"{symbol} - {timeframe}", fontsize=16)
        
        # Price chart
        ax_price = axes[0]
        
        # Plot candlesticks
        self._plot_candlesticks(ax_price, df)
        
        # Plot indicators on price chart
        if indicators:
            self._plot_indicators(ax_price, df, indicators)
        
        # Plot liquidity zones
        if liquidity_zones:
            self._plot_liquidity_zones(ax_price, liquidity_zones)
        
        # Plot trades
        if trades:
            self._plot_trades(ax_price, trades)
        
        # Volume chart
        ax_volume = axes[1]
        self._plot_volume(ax_volume, df)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _plot_candlesticks(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot candlesticks on axis.
        
        Args:
            ax: Matplotlib axis
            df: DataFrame with OHLCV data
        """
        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # Plot candlesticks
        width = 0.6 * (df.index[1] - df.index[0]).total_seconds() / 86400  # 0.6 day width
        
        # Bullish candles (close > open)
        bullish = df[df['close'] > df['open']]
        ax.bar(bullish.index, bullish['close'] - bullish['open'], width, bottom=bullish['open'], color='green', alpha=0.8)
        ax.vlines(bullish.index, bullish['low'], bullish['high'], color='green', linewidth=1)
        
        # Bearish candles (close <= open)
        bearish = df[df['close'] <= df['open']]
        ax.bar(bearish.index, bearish['close'] - bearish['open'], width, bottom=bearish['open'], color='red', alpha=0.8)
        ax.vlines(bearish.index, bearish['low'], bearish['high'], color='red', linewidth=1)
        
        # Set labels
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot volume on axis.
        
        Args:
            ax: Matplotlib axis
            df: DataFrame with OHLCV data
        """
        if 'volume' not in df.columns:
            ax.text(0.5, 0.5, 'Volume data not available', ha='center', va='center')
            return
        
        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # Plot volume bars
        width = 0.6 * (df.index[1] - df.index[0]).total_seconds() / 86400  # 0.6 day width
        
        # Bullish volume (close > open)
        bullish = df[df['close'] > df['open']]
        ax.bar(bullish.index, bullish['volume'], width, color='green', alpha=0.5)
        
        # Bearish volume (close <= open)
        bearish = df[df['close'] <= df['open']]
        ax.bar(bearish.index, bearish['volume'], width, color='red', alpha=0.5)
        
        # Set labels
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
    
    def _plot_indicators(self, ax: plt.Axes, df: pd.DataFrame, indicators: Dict[str, Dict]) -> None:
        """Plot indicators on axis.
        
        Args:
            ax: Matplotlib axis
            df: DataFrame with OHLCV data
            indicators: Dictionary of indicators to plot
        """
        for name, params in indicators.items():
            if name == 'sma' or name == 'ema':
                for period in params.get('periods', []):
                    col_name = f"{name}_{period}"
                    if col_name in df.columns:
                        ax.plot(df.index, df[col_name], label=f"{name.upper()}({period})")
            
            elif name == 'bollinger_bands':
                for period in params.get('periods', []):
                    upper_col = f"bb_upper_{period}"
                    middle_col = f"bb_middle_{period}"
                    lower_col = f"bb_lower_{period}"
                    
                    if upper_col in df.columns and middle_col in df.columns and lower_col in df.columns:
                        ax.plot(df.index, df[upper_col], 'b--', alpha=0.5)
                        ax.plot(df.index, df[middle_col], 'b-', alpha=0.5)
                        ax.plot(df.index, df[lower_col], 'b--', alpha=0.5)
                        ax.fill_between(df.index, df[upper_col], df[lower_col], alpha=0.1, color='blue')
            
            elif name == 'atr_bands':
                for multiplier in params.get('multipliers', []):
                    upper_col = f"atr_upper_{multiplier}"
                    lower_col = f"atr_lower_{multiplier}"
                    
                    if upper_col in df.columns and lower_col in df.columns:
                        ax.plot(df.index, df[upper_col], 'g--', alpha=0.5)
                        ax.plot(df.index, df[lower_col], 'g--', alpha=0.5)
        
        # Add legend
        ax.legend(loc='upper left')
    
    def _plot_liquidity_zones(self, ax: plt.Axes, liquidity_zones: List[Dict]) -> None:
        """Plot liquidity zones on axis.
        
        Args:
            ax: Matplotlib axis
            liquidity_zones: List of liquidity zone dictionaries
        """
        for zone in liquidity_zones:
            start_time = zone.get('start_time')
            end_time = zone.get('end_time', ax.get_xlim()[1])
            price_level = zone.get('price_level')
            zone_type = zone.get('type', 'demand')  # 'demand' or 'supply'
            strength = zone.get('strength', 1)
            
            if not start_time or not price_level:
                continue
            
            # Determine color and alpha based on zone type and strength
            if zone_type.lower() == 'demand':
                color = 'green'
            else:  # supply
                color = 'red'
            
            alpha = min(0.3 + (strength * 0.1), 0.7)  # Stronger zones are more opaque
            
            # Plot zone
            height = price_level * 0.002  # Adjust height based on price level
            rect = plt.Rectangle((mdates.date2num(start_time), price_level - height/2),
                               mdates.date2num(end_time) - mdates.date2num(start_time),
                               height, color=color, alpha=alpha)
            ax.add_patch(rect)
    
    def _plot_trades(self, ax: plt.Axes, trades: List[Dict]) -> None:
        """Plot trades on axis.
        
        Args:
            ax: Matplotlib axis
            trades: List of trade dictionaries
        """
        for trade in trades:
            entry_time = trade.get('entry_time')
            entry_price = trade.get('entry_price')
            exit_time = trade.get('exit_time')
            exit_price = trade.get('exit_price')
            direction = trade.get('direction', 'long')  # 'long' or 'short'
            sl_price = trade.get('sl_price')
            tp_price = trade.get('tp_price')
            
            if not entry_time or not entry_price:
                continue
            
            # Determine color based on direction and result
            if direction.lower() == 'long':
                entry_marker = '^'  # Triangle up
                if exit_price and exit_price > entry_price:
                    color = 'green'  # Profitable long
                else:
                    color = 'red'  # Losing long
            else:  # short
                entry_marker = 'v'  # Triangle down
                if exit_price and exit_price < entry_price:
                    color = 'green'  # Profitable short
                else:
                    color = 'red'  # Losing short
            
            # Plot entry point
            ax.scatter(entry_time, entry_price, marker=entry_marker, color=color, s=100, zorder=5)
            
            # Plot exit point if available
            if exit_time and exit_price:
                ax.scatter(exit_time, exit_price, marker='o', color=color, s=100, zorder=5)
                
                # Connect entry and exit with a line
                ax.plot([entry_time, exit_time], [entry_price, exit_price], color=color, linestyle='-', linewidth=1.5, zorder=4)
            
            # Plot stop loss and take profit levels
            if sl_price:
                ax.axhline(y=sl_price, color='red', linestyle='--', alpha=0.5, xmin=mdates.date2num(entry_time), 
                          xmax=mdates.date2num(exit_time) if exit_time else ax.get_xlim()[1])
            
            if tp_price:
                ax.axhline(y=tp_price, color='green', linestyle='--', alpha=0.5, xmin=mdates.date2num(entry_time),
                          xmax=mdates.date2num(exit_time) if exit_time else ax.get_xlim()[1])


class PerformanceVisualizer:
    """Visualize trading performance metrics."""
    
    def __init__(self, save_dir: str = None):
        """Initialize PerformanceVisualizer.
        
        Args:
            save_dir: Directory to save visualizations (default: logs/performance)
        """
        if save_dir is None:
            self.save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'performance')
        else:
            self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_equity_curve(self, equity_data: pd.Series, initial_balance: float = None, 
                          benchmark: pd.Series = None, save: bool = True) -> plt.Figure:
        """Plot equity curve.
        
        Args:
            equity_data: Series with equity values indexed by datetime
            initial_balance: Initial account balance
            benchmark: Series with benchmark values indexed by datetime
            save: Whether to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(equity_data.index, equity_data.values, label='Strategy', linewidth=2)
        
        # Plot initial balance
        if initial_balance is not None:
            ax.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to start at the same value as equity
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity_data.iloc[0]
            ax.plot(benchmark.index, benchmark_normalized, label='Benchmark', alpha=0.7, linewidth=1.5)
        
        # Set labels and title
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate date labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"equity_curve_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown_chart(self, equity_data: pd.Series, save: bool = True) -> plt.Figure:
        """Plot drawdown chart.
        
        Args:
            equity_data: Series with equity values indexed by datetime
            save: Whether to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown
        rolling_max = equity_data.cummax()
        drawdown = (equity_data - rolling_max) / rolling_max * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color='red', alpha=0.5)
        
        # Set labels and title
        ax.set_title('Drawdown Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate date labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawdown_chart_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns(self, equity_data: pd.Series, save: bool = True) -> plt.Figure:
        """Plot monthly returns heatmap.
        
        Args:
            equity_data: Series with equity values indexed by datetime
            save: Whether to save the plot
        
        Returns:
            Matplotlib figure
        """
        # Calculate daily returns
        daily_returns = equity_data.pct_change().dropna()
        
        # Calculate monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table for the heatmap
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_returns_table = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_returns_table.pivot('Year', 'Month', 'Return')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap=cmap, center=0, ax=ax)
        
        # Set labels and title
        ax.set_title('Monthly Returns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monthly_returns_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_trade_analysis(self, trades: List[Dict], save: bool = True) -> Dict[str, plt.Figure]:
        """Plot trade analysis charts.
        
        Args:
            trades: List of trade dictionaries
            save: Whether to save the plots
        
        Returns:
            Dictionary of Matplotlib figures
        """
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate profit/loss for each trade
        if 'profit_loss' not in trades_df.columns:
            if 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns and 'direction' in trades_df.columns:
                trades_df['profit_loss'] = np.where(
                    trades_df['direction'] == 'long',
                    trades_df['exit_price'] - trades_df['entry_price'],
                    trades_df['entry_price'] - trades_df['exit_price']
                )
        
        # Create figures dictionary
        figures = {}
        
        # 1. Profit/Loss Distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(trades_df['profit_loss'], kde=True, ax=ax1)
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.set_title('Profit/Loss Distribution')
        ax1.set_xlabel('Profit/Loss')
        ax1.set_ylabel('Frequency')
        figures['profit_loss_dist'] = fig1
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profit_loss_dist_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            fig1.savefig(filepath, dpi=150, bbox_inches='tight')
        
        # 2. Win/Loss by Direction
        if 'direction' in trades_df.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            trades_df['result'] = np.where(trades_df['profit_loss'] > 0, 'Win', 'Loss')
            result_by_direction = pd.crosstab(trades_df['direction'], trades_df['result'])
            result_by_direction.plot(kind='bar', ax=ax2)
            ax2.set_title('Win/Loss by Direction')
            ax2.set_xlabel('Direction')
            ax2.set_ylabel('Count')
            ax2.legend(title='Result')
            figures['win_loss_direction'] = fig2
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"win_loss_direction_{timestamp}.png"
                filepath = os.path.join(self.save_dir, filename)
                fig2.savefig(filepath, dpi=150, bbox_inches='tight')
        
        # 3. Profit/Loss Over Time
        if 'exit_time' in trades_df.columns:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            trades_df.sort_values('exit_time', inplace=True)
            trades_df['cumulative_pnl'] = trades_df['profit_loss'].cumsum()
            ax3.plot(trades_df['exit_time'], trades_df['cumulative_pnl'])
            ax3.set_title('Cumulative Profit/Loss Over Time')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Cumulative Profit/Loss')
            ax3.grid(True, alpha=0.3)
            figures['cumulative_pnl'] = fig3
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cumulative_pnl_{timestamp}.png"
                filepath = os.path.join(self.save_dir, filename)
                fig3.savefig(filepath, dpi=150, bbox_inches='tight')
        
        # 4. Trade Duration Analysis
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # hours
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='duration', y='profit_loss', hue='direction', data=trades_df, ax=ax4)
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.set_title('Profit/Loss vs Trade Duration')
            ax4.set_xlabel('Duration (hours)')
            ax4.set_ylabel('Profit/Loss')
            figures['duration_analysis'] = fig4
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"duration_analysis_{timestamp}.png"
                filepath = os.path.join(self.save_dir, filename)
                fig4.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return figures
    
    def plot_performance_metrics(self, metrics: Dict[str, float], save: bool = True) -> plt.Figure:
        """Plot performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            save: Whether to save the plot
        
        Returns:
            Matplotlib figure
        """
        # Select key metrics to display
        key_metrics = {
            'Total Return (%)': metrics.get('total_return_pct', 0),
            'Annualized Return (%)': metrics.get('annualized_return_pct', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
            'Win Rate (%)': metrics.get('win_rate', 0) * 100,
            'Profit Factor': metrics.get('profit_factor', 0),
            'Avg Win/Loss Ratio': metrics.get('avg_win_loss_ratio', 0),
            'Calmar Ratio': metrics.get('calmar_ratio', 0),
            'Recovery Factor': metrics.get('recovery_factor', 0)
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        metrics_df = pd.DataFrame(list(key_metrics.items()), columns=['Metric', 'Value'])
        metrics_df = metrics_df.sort_values('Value')
        
        # Use different colors for positive and negative values
        colors = ['green' if x >= 0 else 'red' for x in metrics_df['Value']]
        
        # Plot horizontal bar chart
        bars = ax.barh(metrics_df['Metric'], metrics_df['Value'], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width - 0.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', ha='left' if width >= 0 else 'right')
        
        # Set labels and title
        ax.set_title('Performance Metrics')
        ax.set_xlabel('Value')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_monte_carlo_simulation(self, simulation_results: np.ndarray, 
                                    original_equity: pd.Series, save: bool = True) -> plt.Figure:
        """Plot Monte Carlo simulation results.
        
        Args:
            simulation_results: 2D array of simulation results [n_simulations, n_periods]
            original_equity: Original equity curve
            save: Whether to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot simulation paths
        for i in range(simulation_results.shape[0]):
            ax.plot(simulation_results[i, :], color='blue', alpha=0.1)
        
        # Plot original equity curve
        ax.plot(original_equity.values, color='red', linewidth=2, label='Original Equity')
        
        # Plot percentiles
        percentiles = np.percentile(simulation_results, [5, 50, 95], axis=0)
        ax.plot(percentiles[1, :], color='green', linewidth=2, label='Median (50th percentile)')
        ax.plot(percentiles[0, :], color='orange', linewidth=1.5, label='5th percentile')
        ax.plot(percentiles[2, :], color='orange', linewidth=1.5, label='95th percentile')
        
        # Fill between percentiles
        ax.fill_between(range(len(percentiles[0, :])), percentiles[0, :], percentiles[2, :], color='orange', alpha=0.2)
        
        # Set labels and title
        ax.set_title('Monte Carlo Simulation')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig