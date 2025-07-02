#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Order Manager module for handling trade execution and management.

This module handles:
- Order placement (market, limit, stop orders)
- Position sizing based on risk parameters
- Order modification and cancellation
- Position tracking and management
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# Import MetaTrader5 module
import MetaTrader5 as mt5

# Import local modules
from config.strategy_params import RiskParams


class OrderManager:
    """Handles order execution and management."""
    
    def __init__(self, risk_params: RiskParams):
        """Initialize OrderManager.
        
        Args:
            risk_params: Risk parameters for position sizing
        """
        self.logger = logging.getLogger('main_logger')
        self.risk_params = risk_params
        self.open_positions = {}
        self.pending_orders = {}
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Position size in lots
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
        
        # Calculate risk amount
        account_balance = account_info.balance
        risk_amount = account_balance * (self.risk_params.risk_per_trade / 100.0)
        
        # Calculate stop loss distance in price units
        if entry_price > stop_loss:  # Long position
            sl_distance = entry_price - stop_loss
        else:  # Short position
            sl_distance = stop_loss - entry_price
        
        # Calculate position size
        point_value = symbol_info.trade_tick_value * (symbol_info.point / symbol_info.trade_tick_size)
        position_size = risk_amount / (sl_distance * point_value)
        
        # Convert to lots
        lot_size = position_size / symbol_info.trade_contract_size
        
        # Round to symbol's lot step
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Apply min/max lot constraints
        lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
        
        # Apply max positions constraint
        if self.risk_params.max_positions > 0:
            open_positions_count = len(self.get_open_positions(symbol))
            if open_positions_count >= self.risk_params.max_positions:
                self.logger.warning(f"Max positions ({self.risk_params.max_positions}) reached for {symbol}")
                return 0.0
        
        # Apply max risk per symbol constraint
        if self.risk_params.max_risk_per_symbol > 0:
            symbol_risk = self.calculate_symbol_risk(symbol)
            max_risk = account_balance * (self.risk_params.max_risk_per_symbol / 100.0)
            if symbol_risk + risk_amount > max_risk:
                self.logger.warning(f"Max risk per symbol ({self.risk_params.max_risk_per_symbol}%) reached for {symbol}")
                return 0.0
        
        return lot_size
    
    def place_order(self, symbol: str, order_type: str, direction: int, volume: float, 
                   price: float, stop_loss: float, take_profit: float, comment: str = "") -> bool:
        """Place a trading order.
        
        Args:
            symbol: Trading symbol
            order_type: Order type ("MARKET", "LIMIT", "STOP")
            direction: Trade direction (1 for buy, -1 for sell)
            volume: Position size in lots
            price: Order price
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
        
        Returns:
            True if order was placed successfully, False otherwise
        """
        if volume <= 0:
            self.logger.error(f"Invalid volume: {volume}")
            return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return False
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_type == "MARKET" else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": self._get_order_type(order_type, direction),
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.risk_params.slippage_points,
            "magic": 123456,  # Magic number for identifying bot orders
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order placement failed: {result.retcode}, {result.comment}")
            return False
        
        # Update positions/orders tracking
        if order_type == "MARKET":
            self._update_positions()
        else:
            self._update_pending_orders()
        
        self.logger.info(f"Order placed: {symbol}, {order_type}, {volume} lots, price: {price}")
        return True
    
    def modify_order(self, ticket: int, price: Optional[float] = None, 
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """Modify an existing order.
        
        Args:
            ticket: Order ticket number
            price: New order price (for pending orders)
            stop_loss: New stop loss price
            take_profit: New take profit price
        
        Returns:
            True if order was modified successfully, False otherwise
        """
        # Check if order exists
        order = mt5.orders_get(ticket=ticket)
        position = mt5.positions_get(ticket=ticket)
        
        if order is None and position is None:
            self.logger.error(f"Order/position {ticket} not found")
            return False
        
        # Prepare modification request
        if order is not None:  # Pending order
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "price": price if price is not None else order[0].price_open,
                "sl": stop_loss if stop_loss is not None else order[0].sl,
                "tp": take_profit if take_profit is not None else order[0].tp,
            }
        else:  # Open position
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": stop_loss if stop_loss is not None else position[0].sl,
                "tp": take_profit if take_profit is not None else position[0].tp,
            }
        
        # Send modification request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order modification failed: {result.retcode}, {result.comment}")
            return False
        
        # Update positions/orders tracking
        self._update_positions()
        self._update_pending_orders()
        
        self.logger.info(f"Order {ticket} modified")
        return True
    
    def cancel_order(self, ticket: int) -> bool:
        """Cancel a pending order.
        
        Args:
            ticket: Order ticket number
        
        Returns:
            True if order was cancelled successfully, False otherwise
        """
        # Check if order exists
        order = mt5.orders_get(ticket=ticket)
        if order is None:
            self.logger.error(f"Order {ticket} not found")
            return False
        
        # Prepare cancellation request
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        
        # Send cancellation request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order cancellation failed: {result.retcode}, {result.comment}")
            return False
        
        # Update pending orders tracking
        self._update_pending_orders()
        
        self.logger.info(f"Order {ticket} cancelled")
        return True
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> bool:
        """Close an open position.
        
        Args:
            ticket: Position ticket number
            volume: Volume to close (if None, close entire position)
        
        Returns:
            True if position was closed successfully, False otherwise
        """
        # Check if position exists
        position = mt5.positions_get(ticket=ticket)
        if position is None:
            self.logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        # Determine volume to close
        close_volume = volume if volume is not None else position.volume
        if close_volume > position.volume:
            self.logger.error(f"Close volume {close_volume} exceeds position volume {position.volume}")
            return False
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": self.risk_params.slippage_points,
            "magic": 123456,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send close request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Position close failed: {result.retcode}, {result.comment}")
            return False
        
        # Update positions tracking
        self._update_positions()
        
        self.logger.info(f"Position {ticket} closed")
        return True
    
    def close_all_positions(self, symbol: Optional[str] = None) -> bool:
        """Close all open positions.
        
        Args:
            symbol: Symbol to close positions for (if None, close all positions)
        
        Returns:
            True if all positions were closed successfully, False otherwise
        """
        # Get open positions
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return True  # No positions to close
        
        # Close each position
        success = True
        for position in positions:
            if not self.close_position(position.ticket):
                success = False
        
        return success
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open positions.
        
        Args:
            symbol: Symbol to get positions for (if None, get all positions)
        
        Returns:
            List of open positions
        """
        # Update positions tracking
        self._update_positions()
        
        # Filter positions by symbol if specified
        if symbol:
            return [pos for pos in self.open_positions.values() if pos['symbol'] == symbol]
        else:
            return list(self.open_positions.values())
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get pending orders.
        
        Args:
            symbol: Symbol to get orders for (if None, get all orders)
        
        Returns:
            List of pending orders
        """
        # Update orders tracking
        self._update_pending_orders()
        
        # Filter orders by symbol if specified
        if symbol:
            return [order for order in self.pending_orders.values() if order['symbol'] == symbol]
        else:
            return list(self.pending_orders.values())
    
    def calculate_symbol_risk(self, symbol: str) -> float:
        """Calculate current risk for a symbol.
        
        Args:
            symbol: Symbol to calculate risk for
        
        Returns:
            Risk amount in account currency
        """
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return 0.0
        
        # Get open positions for symbol
        positions = self.get_open_positions(symbol)
        
        # Calculate total risk
        total_risk = 0.0
        for position in positions:
            # Calculate risk for this position
            if position['sl'] > 0:  # If stop loss is set
                if position['type'] == mt5.ORDER_TYPE_BUY:  # Long position
                    risk_per_lot = (position['price_open'] - position['sl']) * position['point_value']
                else:  # Short position
                    risk_per_lot = (position['sl'] - position['price_open']) * position['point_value']
                
                position_risk = risk_per_lot * position['volume']
                total_risk += position_risk
        
        return total_risk
    
    def _get_order_type(self, order_type: str, direction: int) -> int:
        """Get MT5 order type constant.
        
        Args:
            order_type: Order type ("MARKET", "LIMIT", "STOP")
            direction: Trade direction (1 for buy, -1 for sell)
        
        Returns:
            MT5 order type constant
        """
        if order_type == "MARKET":
            return mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        elif order_type == "LIMIT":
            return mt5.ORDER_TYPE_BUY_LIMIT if direction == 1 else mt5.ORDER_TYPE_SELL_LIMIT
        elif order_type == "STOP":
            return mt5.ORDER_TYPE_BUY_STOP if direction == 1 else mt5.ORDER_TYPE_SELL_STOP
        else:
            self.logger.error(f"Unknown order type: {order_type}")
            return mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    
    def _update_positions(self) -> None:
        """Update open positions tracking."""
        # Get all open positions
        positions = mt5.positions_get()
        if positions is None:
            self.open_positions = {}
            return
        
        # Update positions dictionary
        new_positions = {}
        for position in positions:
            # Get symbol info
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                continue
            
            # Calculate point value
            point_value = symbol_info.trade_tick_value * (symbol_info.point / symbol_info.trade_tick_size)
            
            # Add position to dictionary
            new_positions[position.ticket] = {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': position.type,
                'volume': position.volume,
                'price_open': position.price_open,
                'sl': position.sl,
                'tp': position.tp,
                'profit': position.profit,
                'swap': position.swap,
                'magic': position.magic,
                'comment': position.comment,
                'time': position.time,
                'point_value': point_value,
            }
        
        self.open_positions = new_positions
    
    def _update_pending_orders(self) -> None:
        """Update pending orders tracking."""
        # Get all pending orders
        orders = mt5.orders_get()
        if orders is None:
            self.pending_orders = {}
            return
        
        # Update orders dictionary
        new_orders = {}
        for order in orders:
            # Get symbol info
            symbol_info = mt5.symbol_info(order.symbol)
            if symbol_info is None:
                continue
            
            # Add order to dictionary
            new_orders[order.ticket] = {
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': order.type,
                'volume': order.volume_current,
                'price_open': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'magic': order.magic,
                'comment': order.comment,
                'time_setup': order.time_setup,
                'time_expiration': order.time_expiration,
            }
        
        self.pending_orders = new_orders