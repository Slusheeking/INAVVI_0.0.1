"""
Dollar Profit Objective

This module provides custom objective functions for XGBoost that optimize for dollar profit
rather than just prediction accuracy. This is a critical component for trading systems
where the magnitude of profit/loss matters more than binary classification accuracy.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)

def dollar_profit_objective(
    predt: np.ndarray, 
    dtrain: Any, 
    position_size: float = 1000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_aversion: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom objective function for XGBoost that optimizes for dollar profit.
    
    This function calculates the gradient and hessian for XGBoost optimization
    based on the expected dollar profit/loss of trades, accounting for:
    - Position sizing
    - Transaction costs
    - Slippage
    - Risk aversion
    
    Args:
        predt: Predicted values (probabilities)
        dtrain: XGBoost DMatrix containing training data
        position_size: Base position size in dollars
        transaction_cost: Transaction cost as a fraction of position size
        slippage: Expected slippage as a fraction of position size
        risk_aversion: Risk aversion parameter (higher values penalize losses more)
        
    Returns:
        Tuple of gradient and hessian arrays for XGBoost optimization
    """
    # Get actual returns from the training data
    y = dtrain.get_label()
    
    # Get auxiliary data if available (price, volatility, etc.)
    try:
        aux_data = dtrain.get_float_info('aux_data')
        price = aux_data[:, 0] if aux_data.shape[1] > 0 else np.ones_like(y)
        volatility = aux_data[:, 1] if aux_data.shape[1] > 1 else np.ones_like(y) * 0.01
    except Exception:
        # If auxiliary data is not available, use default values
        price = np.ones_like(y)
        volatility = np.ones_like(y) * 0.01
    
    # Calculate position sizes based on prediction confidence and volatility
    # Higher confidence and lower volatility lead to larger positions
    confidence = np.abs(predt - 0.5) * 2  # Scale to [0, 1]
    vol_adjustment = 1.0 / (1.0 + volatility * 10)  # Lower volatility -> higher adjustment
    adjusted_position_size = position_size * confidence * vol_adjustment
    
    # Calculate expected profit/loss
    # For long positions (predt > 0.5), profit = position_size * return
    # For short positions (predt < 0.5), profit = position_size * -return
    position_direction = np.sign(predt - 0.5)
    expected_return = y * position_direction
    
    # Calculate transaction costs and slippage
    costs = (transaction_cost + slippage) * adjusted_position_size
    
    # Calculate dollar profit/loss
    dollar_profit = adjusted_position_size * expected_return - costs
    
    # Apply risk aversion (penalize losses more than gains)
    risk_adjusted_profit = np.where(
        dollar_profit >= 0,
        dollar_profit,
        dollar_profit * risk_aversion
    )
    
    # Calculate gradient (negative because XGBoost minimizes the objective)
    # The gradient is the derivative of the dollar profit with respect to the prediction
    gradient = -risk_adjusted_profit * position_size * np.sign(predt - 0.5)
    
    # Calculate hessian (second derivative)
    # For simplicity, we use a constant hessian
    hessian = np.ones_like(gradient) * position_size
    
    return gradient, hessian

def create_dollar_profit_objective(
    position_size: float = 1000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_aversion: float = 1.0
) -> Callable:
    """
    Create a dollar profit objective function with fixed parameters.
    
    Args:
        position_size: Base position size in dollars
        transaction_cost: Transaction cost as a fraction of position size
        slippage: Expected slippage as a fraction of position size
        risk_aversion: Risk aversion parameter (higher values penalize losses more)
        
    Returns:
        Dollar profit objective function with fixed parameters
    """
    def objective_fn(predt: np.ndarray, dtrain: Any) -> Tuple[np.ndarray, np.ndarray]:
        return dollar_profit_objective(
            predt, 
            dtrain, 
            position_size=position_size,
            transaction_cost=transaction_cost,
            slippage=slippage,
            risk_aversion=risk_aversion
        )
    
    return objective_fn

def dollar_profit_eval_metric(
    predt: np.ndarray, 
    dtrain: Any,
    position_size: float = 1000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> Tuple[str, float]:
    """
    Evaluation metric for XGBoost that calculates expected dollar profit.
    
    Args:
        predt: Predicted values (probabilities)
        dtrain: XGBoost DMatrix containing training data
        position_size: Base position size in dollars
        transaction_cost: Transaction cost as a fraction of position size
        slippage: Expected slippage as a fraction of position size
        
    Returns:
        Tuple of metric name and value
    """
    # Get actual returns from the training data
    y = dtrain.get_label()
    
    # Get auxiliary data if available (price, volatility, etc.)
    try:
        aux_data = dtrain.get_float_info('aux_data')
        price = aux_data[:, 0] if aux_data.shape[1] > 0 else np.ones_like(y)
        volatility = aux_data[:, 1] if aux_data.shape[1] > 1 else np.ones_like(y) * 0.01
    except Exception:
        # If auxiliary data is not available, use default values
        price = np.ones_like(y)
        volatility = np.ones_like(y) * 0.01
    
    # Calculate position sizes based on prediction confidence and volatility
    confidence = np.abs(predt - 0.5) * 2  # Scale to [0, 1]
    vol_adjustment = 1.0 / (1.0 + volatility * 10)  # Lower volatility -> higher adjustment
    adjusted_position_size = position_size * confidence * vol_adjustment
    
    # Calculate expected profit/loss
    position_direction = np.sign(predt - 0.5)
    expected_return = y * position_direction
    
    # Calculate transaction costs and slippage
    costs = (transaction_cost + slippage) * adjusted_position_size
    
    # Calculate dollar profit/loss
    dollar_profit = adjusted_position_size * expected_return - costs
    
    # Calculate total profit
    total_profit = np.sum(dollar_profit)
    
    # Calculate profit per trade
    profit_per_trade = total_profit / len(y)
    
    return 'dollar_profit', profit_per_trade

def create_dollar_profit_eval_metric(
    position_size: float = 1000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005
) -> Callable:
    """
    Create a dollar profit evaluation metric with fixed parameters.
    
    Args:
        position_size: Base position size in dollars
        transaction_cost: Transaction cost as a fraction of position size
        slippage: Expected slippage as a fraction of position size
        
    Returns:
        Dollar profit evaluation metric with fixed parameters
    """
    def eval_metric(predt: np.ndarray, dtrain: Any) -> Tuple[str, float]:
        return dollar_profit_eval_metric(
            predt, 
            dtrain, 
            position_size=position_size,
            transaction_cost=transaction_cost,
            slippage=slippage
        )
    
    return eval_metric

def sharpe_ratio_eval_metric(
    predt: np.ndarray, 
    dtrain: Any,
    position_size: float = 1000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0
) -> Tuple[str, float]:
    """
    Evaluation metric for XGBoost that calculates the Sharpe ratio.
    
    Args:
        predt: Predicted values (probabilities)
        dtrain: XGBoost DMatrix containing training data
        position_size: Base position size in dollars
        transaction_cost: Transaction cost as a fraction of position size
        slippage: Expected slippage as a fraction of position size
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Tuple of metric name and value
    """
    # Get actual returns from the training data
    y = dtrain.get_label()
    
    # Calculate position sizes based on prediction confidence
    confidence = np.abs(predt - 0.5) * 2  # Scale to [0, 1]
    adjusted_position_size = position_size * confidence
    
    # Calculate expected profit/loss
    position_direction = np.sign(predt - 0.5)
    expected_return = y * position_direction
    
    # Calculate transaction costs and slippage
    costs = (transaction_cost + slippage) * adjusted_position_size
    
    # Calculate dollar profit/loss
    dollar_profit = adjusted_position_size * expected_return - costs
    
    # Calculate mean and standard deviation of profits
    mean_profit = np.mean(dollar_profit)
    std_profit = np.std(dollar_profit)
    
    # Calculate Sharpe ratio
    # Avoid division by zero
    if std_profit == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (mean_profit - risk_free_rate) / std_profit
    
    return 'sharpe_ratio', sharpe_ratio

def create_sharpe_ratio_eval_metric(
    position_size: float = 1000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.0
) -> Callable:
    """
    Create a Sharpe ratio evaluation metric with fixed parameters.
    
    Args:
        position_size: Base position size in dollars
        transaction_cost: Transaction cost as a fraction of position size
        slippage: Expected slippage as a fraction of position size
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio evaluation metric with fixed parameters
    """
    def eval_metric(predt: np.ndarray, dtrain: Any) -> Tuple[str, float]:
        return sharpe_ratio_eval_metric(
            predt, 
            dtrain, 
            position_size=position_size,
            transaction_cost=transaction_cost,
            slippage=slippage,
            risk_free_rate=risk_free_rate
        )
    
    return eval_metric