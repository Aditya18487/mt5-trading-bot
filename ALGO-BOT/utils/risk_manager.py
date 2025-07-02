#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Management module for the trading bot.

This module handles:
- Risk assessment for trades
- Position sizing calculations
- Exposure management
- Drawdown monitoring and protection
- Trading session management
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time, timedelta
import MetaTrader5 as mt5

from config.strategy_params import RiskParams


class RiskManager:
    """Handles risk assessment and management."""
    
    def __init__(self, risk_params: RiskParams):
        """Initialize RiskManager.
        
        Args:
            risk_params: Risk parameters for risk management
        """
        self.logger = logging.getLogger('main_logger')
        self.risk_params = risk_params
        self.daily_profit_loss = 0.0
        self.weekly_profit_loss = 0.0
        self.monthly_profit_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.last_balance_check = datetime.now()
        
        # Initialize peak balance
        account_info = mt5.account_info()
        if account_info:
            self.peak_balance = account_info.balance
    
    def update_metrics(self) -> None:
        """Update risk metrics based on current account state."""
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return
        
        current_balance = account_info.balance
        current_time = datetime.now()
        
        # Update peak balance and drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        else:
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # Update profit/loss metrics if enough time has passed
        if (current_time - self.last_balance_check).total_seconds() >= 60:  # Update every minute
            # Get history orders for today
            from_date = datetime(current_time.year, current_time.month, current_time.day)
            history_orders = mt5.history_deals_get(from_date, current_time)
            
            if history_orders is not None:
                # Calculate daily P/L
                self.daily_profit_loss = sum(deal.profit for deal in history_orders)
                
                # Calculate weekly P/L (if needed)
                if current_time.weekday() == 0 or self.weekly_profit_loss == 0:  # Monday or not initialized
                    # Get history from beginning of week
                    days_to_subtract = current_time.weekday()
                    from_week = datetime(current_time.year, current_time.month, current_time.day) - \
                                 timedelta(days=days_to_subtract)
                    week_orders = mt5.history_deals_get(from_week, current_time)
                    if week_orders is not None:
                        self.weekly_profit_loss = sum(deal.profit for deal in week_orders)
                
                # Calculate monthly P/L (if needed)
                if current_time.day == 1 or self.monthly_profit_loss == 0:  # First day of month or not initialized
                    # Get history from beginning of month
                    from_month = datetime(current_time.year, current_time.month, 1)
                    month_orders = mt5.history_deals_get(from_month, current_time)
                    if month_orders is not None:
                        self.monthly_profit_loss = sum(deal.profit for deal in month_orders)
            
            self.last_balance_check = current_time
    
    def check_trading_allowed(self, symbol: str) -> bool:
        """Check if trading is allowed based on risk parameters.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            True if trading is allowed, False otherwise
        """
        # Update risk metrics
        self.update_metrics()
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return False
        
        current_balance = account_info.balance
        
        # Check max drawdown
        if self.risk_params.max_drawdown > 0 and self.max_drawdown >= self.risk_params.max_drawdown:
            self.logger.warning(f"Max drawdown ({self.risk_params.max_drawdown}%) reached: {self.max_drawdown:.2f}%")
            return False
        
        # Check daily loss limit
        if self.risk_params.daily_loss_limit > 0:
            daily_loss_amount = current_balance * (self.risk_params.daily_loss_limit / 100.0)
            if self.daily_profit_loss <= -daily_loss_amount:
                self.logger.warning(f"Daily loss limit ({self.risk_params.daily_loss_limit}%) reached")
                return False
        
        # Check weekly loss limit
        if self.risk_params.weekly_loss_limit > 0:
            weekly_loss_amount = current_balance * (self.risk_params.weekly_loss_limit / 100.0)
            if self.weekly_profit_loss <= -weekly_loss_amount:
                self.logger.warning(f"Weekly loss limit ({self.risk_params.weekly_loss_limit}%) reached")
                return False
        
        # Check monthly loss limit
        if self.risk_params.monthly_loss_limit > 0:
            monthly_loss_amount = current_balance * (self.risk_params.monthly_loss_limit / 100.0)
            if self.monthly_profit_loss <= -monthly_loss_amount:
                self.logger.warning(f"Monthly loss limit ({self.risk_params.monthly_loss_limit}%) reached")
                return False
        
        # Check trading hours
        if not self._is_within_trading_hours():
            self.logger.info("Outside of allowed trading hours")
            return False
        
        # Check max open positions
        if self.risk_params.max_positions > 0:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if positions is not None and len(positions) >= self.risk_params.max_positions:
                self.logger.warning(f"Max positions ({self.risk_params.max_positions}) reached")
                return False
        
        # Check max risk per symbol
        if symbol and self.risk_params.max_risk_per_symbol > 0:
            symbol_positions = mt5.positions_get(symbol=symbol)
            if symbol_positions is not None:
                symbol_risk = sum(self._calculate_position_risk(pos) for pos in symbol_positions)
                max_risk = current_balance * (self.risk_params.max_risk_per_symbol / 100.0)
                if symbol_risk >= max_risk:
                    self.logger.warning(f"Max risk per symbol ({self.risk_params.max_risk_per_symbol}%) reached for {symbol}")
                    return False
        
        return True
    
    def check_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        """Check if risk-reward ratio meets minimum requirement.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            True if risk-reward ratio is acceptable, False otherwise
        """
        # Calculate risk and reward
        if entry_price > stop_loss:  # Long position
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # Short position
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        # Calculate risk-reward ratio
        if risk <= 0:
            self.logger.error("Invalid risk (zero or negative)")
            return False
        
        rr_ratio = reward / risk
        
        # Check against minimum requirement
        if rr_ratio < self.risk_params.min_risk_reward_ratio:
            self.logger.warning(f"Risk-reward ratio ({rr_ratio:.2f}) below minimum ({self.risk_params.min_risk_reward_ratio})")
            return False
        
        return True
    
    def adjust_position_size(self, symbol: str, calculated_size: float) -> float:
        """Adjust position size based on risk parameters and market conditions.
        
        Args:
            symbol: Trading symbol
            calculated_size: Initially calculated position size
        
        Returns:
            Adjusted position size
        """
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return 0.0
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return 0.0
        
        # Start with calculated size
        adjusted_size = calculated_size
        
        # Apply max position size limit
        if self.risk_params.max_position_size > 0:
            max_size = self.risk_params.max_position_size
            adjusted_size = min(adjusted_size, max_size)
        
        # Apply volatility-based adjustment if ATR is available
        # This would require ATR data from market_data module
        # For now, we'll skip this adjustment
        
        # Apply drawdown-based adjustment
        if self.max_drawdown > 0 and self.risk_params.drawdown_adjustment_factor > 0:
            # Reduce position size as drawdown increases
            drawdown_factor = 1.0 - (self.max_drawdown / 100.0 * self.risk_params.drawdown_adjustment_factor)
            drawdown_factor = max(0.1, drawdown_factor)  # Don't reduce below 10%
            adjusted_size *= drawdown_factor
        
        # Apply consecutive losses adjustment
        consecutive_losses = self._get_consecutive_losses(symbol)
        if consecutive_losses >= self.risk_params.consecutive_losses_threshold:
            # Reduce position size after consecutive losses
            loss_factor = 1.0 - (self.risk_params.consecutive_losses_factor * 
                               (consecutive_losses - self.risk_params.consecutive_losses_threshold + 1))
            loss_factor = max(0.1, loss_factor)  # Don't reduce below 10%
            adjusted_size *= loss_factor
        
        # Round to symbol's lot step
        lot_step = symbol_info.volume_step
        adjusted_size = math.floor(adjusted_size / lot_step) * lot_step
        
        # Apply min/max lot constraints
        adjusted_size = max(symbol_info.volume_min, min(symbol_info.volume_max, adjusted_size))
        
        return adjusted_size
    
    def _is_within_trading_hours(self) -> bool:
        """Check if current time is within allowed trading hours.
        
        Returns:
            True if within trading hours, False otherwise
        """
        current_time = datetime.now().time()
        
        # If no trading hours are specified, allow trading at any time
        if not self.risk_params.trading_hours_start or not self.risk_params.trading_hours_end:
            return True
        
        # Parse trading hours
        try:
            start_hour, start_minute = map(int, self.risk_params.trading_hours_start.split(':'))
            end_hour, end_minute = map(int, self.risk_params.trading_hours_end.split(':'))
            
            start_time = time(start_hour, start_minute)
            end_time = time(end_hour, end_minute)
            
            # Check if current time is within range
            if start_time <= end_time:
                # Normal case (e.g., 9:00-17:00)
                return start_time <= current_time <= end_time
            else:
                # Overnight case (e.g., 22:00-6:00)
                return current_time >= start_time or current_time <= end_time
        except ValueError:
            self.logger.error("Invalid trading hours format")
            return True  # Allow trading if format is invalid
    
    def _calculate_position_risk(self, position) -> float:
        """Calculate risk for a position.
        
        Args:
            position: Position object from MT5
        
        Returns:
            Risk amount in account currency
        """
        # Get symbol info
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            return 0.0
        
        # Calculate point value
        point_value = symbol_info.trade_tick_value * (symbol_info.point / symbol_info.trade_tick_size)
        
        # Calculate risk
        if position.sl > 0:  # If stop loss is set
            if position.type == mt5.ORDER_TYPE_BUY:  # Long position
                risk_per_lot = (position.price_open - position.sl) * point_value
            else:  # Short position
                risk_per_lot = (position.sl - position.price_open) * point_value
            
            return risk_per_lot * position.volume
        else:
            # If no stop loss, assume risk is a percentage of position value
            position_value = position.price_open * position.volume * symbol_info.trade_contract_size
            return position_value * 0.02  # Assume 2% risk if no stop loss
    
    def _get_consecutive_losses(self, symbol: Optional[str] = None) -> int:
        """Get number of consecutive losses.
        
        Args:
            symbol: Symbol to check (if None, check all symbols)
        
        Returns:
            Number of consecutive losses
        """
        # Get recent deals
        from_date = datetime.now() - timedelta(days=7)  # Look back 7 days
        history_deals = mt5.history_deals_get(from_date, datetime.now())
        
        if history_deals is None:
            return 0
        
        # Filter by symbol if specified
        if symbol:
            deals = [deal for deal in history_deals if deal.symbol == symbol]
        else:
            deals = history_deals
        
        # Sort by time
        deals = sorted(deals, key=lambda x: x.time, reverse=True)
        
        # Count consecutive losses
        count = 0
        for deal in deals:
            if deal.profit < 0:
                count += 1
            else:
                break  # Stop counting when a profitable deal is found
        
        return count


class TradeSession:
    """Manages trading sessions and time-based risk adjustments."""
    
    def __init__(self, risk_params: RiskParams):
        """Initialize TradeSession.
        
        Args:
            risk_params: Risk parameters for session management
        """
        self.logger = logging.getLogger('main_logger')
        self.risk_params = risk_params
        self.session_start = datetime.now()
        self.session_trades = 0
        self.session_profit_loss = 0.0
        self.session_win_count = 0
        self.session_loss_count = 0
    
    def start_session(self) -> None:
        """Start a new trading session."""
        self.session_start = datetime.now()
        self.session_trades = 0
        self.session_profit_loss = 0.0
        self.session_win_count = 0
        self.session_loss_count = 0
        self.logger.info(f"Trading session started at {self.session_start}")
    
    def end_session(self) -> Dict:
        """End the current trading session.
        
        Returns:
            Session statistics
        """
        session_end = datetime.now()
        session_duration = session_end - self.session_start
        
        # Calculate win rate
        win_rate = 0.0
        if self.session_trades > 0:
            win_rate = (self.session_win_count / self.session_trades) * 100
        
        # Log session summary
        self.logger.info(f"Trading session ended at {session_end}")
        self.logger.info(f"Session duration: {session_duration}")
        self.logger.info(f"Total trades: {self.session_trades}")
        self.logger.info(f"Win/Loss: {self.session_win_count}/{self.session_loss_count}")
        self.logger.info(f"Win rate: {win_rate:.2f}%")
        self.logger.info(f"Profit/Loss: {self.session_profit_loss:.2f}")
        
        # Return session statistics
        return {
            'start_time': self.session_start,
            'end_time': session_end,
            'duration': session_duration,
            'total_trades': self.session_trades,
            'wins': self.session_win_count,
            'losses': self.session_loss_count,
            'win_rate': win_rate,
            'profit_loss': self.session_profit_loss,
        }
    
    def update_session_stats(self, profit: float) -> None:
        """Update session statistics after a trade.
        
        Args:
            profit: Profit/loss from the trade
        """
        self.session_trades += 1
        self.session_profit_loss += profit
        
        if profit > 0:
            self.session_win_count += 1
        else:
            self.session_loss_count += 1
    
    def should_continue_trading(self) -> bool:
        """Check if trading should continue in the current session.
        
        Returns:
            True if trading should continue, False otherwise
        """
        # Check max trades per session
        if self.risk_params.max_trades_per_session > 0 and self.session_trades >= self.risk_params.max_trades_per_session:
            self.logger.info(f"Max trades per session ({self.risk_params.max_trades_per_session}) reached")
            return False
        
        # Check session profit target
        if self.risk_params.session_profit_target > 0 and self.session_profit_loss >= self.risk_params.session_profit_target:
            self.logger.info(f"Session profit target ({self.risk_params.session_profit_target}) reached")
            return False
        
        # Check session loss limit
        if self.risk_params.session_loss_limit > 0 and self.session_profit_loss <= -self.risk_params.session_loss_limit:
            self.logger.info(f"Session loss limit ({self.risk_params.session_loss_limit}) reached")
            return False
        
        # Check consecutive losses
        if self.risk_params.session_max_consecutive_losses > 0:
            consecutive_losses = self._get_session_consecutive_losses()
            if consecutive_losses >= self.risk_params.session_max_consecutive_losses:
                self.logger.info(f"Session max consecutive losses ({self.risk_params.session_max_consecutive_losses}) reached")
                return False
        
        # Check session duration
        if self.risk_params.max_session_duration_hours > 0:
            session_duration = datetime.now() - self.session_start
            max_duration = timedelta(hours=self.risk_params.max_session_duration_hours)
            if session_duration >= max_duration:
                self.logger.info(f"Max session duration ({self.risk_params.max_session_duration_hours} hours) reached")
                return False
        
        return True
    
    def _get_session_consecutive_losses(self) -> int:
        """Get number of consecutive losses in the current session.
        
        Returns:
            Number of consecutive losses
        """
        # Get recent deals since session start
        history_deals = mt5.history_deals_get(self.session_start, datetime.now())
        
        if history_deals is None:
            return 0
        
        # Sort by time
        deals = sorted(history_deals, key=lambda x: x.time, reverse=True)
        
        # Count consecutive losses
        count = 0
        for deal in deals:
            if deal.profit < 0:
                count += 1
            else:
                break  # Stop counting when a profitable deal is found
        
        return count