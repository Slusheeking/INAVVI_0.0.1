# Risk Management and Position Sizing

This document details how the Autonomous Trading System manages risk and determines position sizes for trades.

## Risk Management Framework

The risk management framework operates at multiple levels to ensure robust protection against excessive losses:

### 1. Portfolio-Level Risk Management

The `portfolio_risk_manager.py` component enforces portfolio-wide risk constraints:

```python
# From src/trading_strategy/risk/portfolio_risk_manager.py
class PortfolioRiskManager:
    def __init__(self, config):
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.02)  # 2% daily VaR
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.10)  # 10% max drawdown
        self.sector_exposure_limits = config.get("sector_exposure_limits", {})
        self.asset_class_limits = config.get("asset_class_limits", {})
        
    def check_portfolio_risk(self, current_positions, proposed_trades):
        """Check if proposed trades would exceed portfolio risk limits."""
        # Calculate Value at Risk (VaR)
        # Check drawdown limits
        # Verify sector exposure
        # Ensure asset class diversification
```

Key portfolio risk metrics tracked:
- **Value at Risk (VaR)**: 95% confidence daily VaR
- **Expected Shortfall**: Average loss beyond VaR
- **Maximum Drawdown**: Peak-to-trough decline
- **Beta Exposure**: Market sensitivity
- **Sector Concentration**: Exposure by sector
- **Asset Class Diversification**: Allocation across asset classes

### 2. Strategy-Level Risk Management

Each trading strategy has its own risk parameters:

```python
# Example strategy risk configuration
STRATEGY_RISK_CONFIG = {
    "momentum_strategy": {
        "max_positions": 20,
        "max_correlation": 0.7,
        "stop_loss_percentage": 0.05,
        "take_profit_percentage": 0.15
    },
    "mean_reversion_strategy": {
        "max_positions": 15,
        "max_correlation": 0.5,
        "stop_loss_percentage": 0.03,
        "take_profit_percentage": 0.08
    }
}
```

Strategy-specific risk controls include:
- **Maximum Number of Positions**: Limits the number of concurrent positions
- **Correlation Limits**: Prevents excessive correlation between positions
- **Strategy-Specific Stop Losses**: Customized stop-loss levels by strategy
- **Maximum Drawdown Thresholds**: Strategy-specific drawdown limits
- **Time-Based Exit Rules**: Maximum holding periods for positions

### 3. Position-Level Risk Management

Individual positions have their own risk controls:

```python
# From src/trading_strategy/risk/stop_loss_manager.py
class StopLossManager:
    def __init__(self, config):
        self.default_stop_loss_pct = config.get("default_stop_loss_pct", 0.02)
        self.trailing_stop_activation_pct = config.get("trailing_stop_activation_pct", 0.01)
        self.trailing_stop_distance_pct = config.get("trailing_stop_distance_pct", 0.01)
        
    def calculate_stop_loss(self, entry_price, position_type, volatility=None, atr=None):
        """Calculate stop loss price for a position."""
        if position_type == "long":
            # For long positions, stop loss is below entry price
            if atr is not None:
                # Use ATR-based stop loss if available
                stop_distance = atr * 2
            else:
                # Use percentage-based stop loss
                stop_distance = entry_price * self.default_stop_loss_pct
                
            return entry_price - stop_distance
        else:
            # For short positions, stop loss is above entry price
            if atr is not None:
                stop_distance = atr * 2
            else:
                stop_distance = entry_price * self.default_stop_loss_pct
                
            return entry_price + stop_distance
```

Position-level risk controls include:
- **Stop Losses**: Fixed, percentage-based, or volatility-adjusted
- **Trailing Stops**: Dynamic stops that move with favorable price movement
- **Take Profit Levels**: Predetermined profit targets
- **Position Sizing**: Risk-based position sizing (detailed below)
- **Time Stops**: Exit positions after a specified time period

## Position Sizing Methodologies

The system uses several position sizing methodologies depending on the strategy and market conditions:

### 1. Fixed Risk Position Sizing

This method risks a fixed percentage of the portfolio on each trade:

```python
# From src/trading_strategy/sizing/risk_based_position_sizer.py
def calculate_position_size_fixed_risk(account_value, risk_per_trade, entry_price, stop_loss_price):
    """Calculate position size based on fixed risk percentage.
    
    Args:
        account_value: Total account value
        risk_per_trade: Percentage of account to risk per trade (e.g., 0.01 for 1%)
        entry_price: Entry price for the position
        stop_loss_price: Stop loss price for the position
        
    Returns:
        Position size in number of shares/contracts
    """
    # Calculate dollar risk
    dollar_risk = account_value * risk_per_trade
    
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss_price)
    
    # Calculate position size
    if risk_per_share > 0:
        position_size = dollar_risk / risk_per_share
    else:
        position_size = 0
        
    return position_size
```

Key parameters:
- **Risk Per Trade**: Typically 0.5% to 2% of account value
- **Stop Loss Distance**: Determines risk per share/contract
- **Account Value**: Total portfolio value

### 2. Volatility-Adjusted Position Sizing

This method adjusts position size based on market volatility:

```python
# From src/trading_strategy/sizing/risk_based_position_sizer.py
def calculate_position_size_volatility_adjusted(account_value, risk_per_trade, atr, atr_multiplier=2):
    """Calculate position size based on volatility (ATR).
    
    Args:
        account_value: Total account value
        risk_per_trade: Percentage of account to risk per trade
        atr: Average True Range (measure of volatility)
        atr_multiplier: Multiplier for ATR to determine stop distance
        
    Returns:
        Position size in number of shares/contracts
    """
    # Calculate dollar risk
    dollar_risk = account_value * risk_per_trade
    
    # Calculate risk per share based on ATR
    risk_per_share = atr * atr_multiplier
    
    # Calculate position size
    if risk_per_share > 0:
        position_size = dollar_risk / risk_per_share
    else:
        position_size = 0
        
    return position_size
```

Key parameters:
- **ATR (Average True Range)**: Measure of market volatility
- **ATR Multiplier**: Typically 2-3x ATR for stop distance
- **Risk Per Trade**: Adjusted based on market conditions

### 3. Kelly Criterion Position Sizing

This method uses the Kelly Criterion for optimal position sizing:

```python
# From src/trading_strategy/sizing/risk_based_position_sizer.py
def calculate_position_size_kelly(win_rate, win_loss_ratio, kelly_fraction=0.5):
    """Calculate position size using the Kelly Criterion.
    
    Args:
        win_rate: Historical win rate (0-1)
        win_loss_ratio: Ratio of average win to average loss
        kelly_fraction: Fraction of full Kelly to use (typically 0.5 for Half Kelly)
        
    Returns:
        Position size as a fraction of account
    """
    # Calculate full Kelly percentage
    kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # Apply Kelly fraction (Half Kelly is common in practice)
    fractional_kelly = kelly_percentage * kelly_fraction
    
    # Ensure non-negative position size
    return max(0, fractional_kelly)
```

Key parameters:
- **Win Rate**: Historical probability of winning trades
- **Win/Loss Ratio**: Ratio of average win to average loss
- **Kelly Fraction**: Typically 0.5 (Half Kelly) for conservative sizing

### 4. Portfolio Optimization Position Sizing

This method uses modern portfolio theory for position sizing:

```python
# From src/trading_strategy/sizing/portfolio_allocator.py
def calculate_optimal_portfolio_weights(expected_returns, covariance_matrix, risk_aversion=1.0):
    """Calculate optimal portfolio weights using mean-variance optimization.
    
    Args:
        expected_returns: Vector of expected returns for each asset
        covariance_matrix: Covariance matrix of asset returns
        risk_aversion: Risk aversion parameter (higher = more conservative)
        
    Returns:
        Array of optimal portfolio weights
    """
    # Calculate optimal weights using mean-variance optimization
    # w* = (1/λ) * Σ^(-1) * μ
    # where λ is risk aversion, Σ is covariance matrix, μ is expected returns
    
    inv_covariance = np.linalg.inv(covariance_matrix)
    weights = (1.0 / risk_aversion) * np.dot(inv_covariance, expected_returns)
    
    # Normalize weights to sum to 1.0
    weights = weights / np.sum(np.abs(weights))
    
    return weights
```

Key parameters:
- **Expected Returns**: Forecasted returns for each asset
- **Covariance Matrix**: Measures relationships between assets
- **Risk Aversion**: Controls trade-off between risk and return

## Position Sizing Constraints

All position sizes are subject to additional constraints:

### 1. Maximum Position Size

```python
# From src/trading_strategy/sizing/position_sizer.py
def apply_position_size_constraints(position_size, max_position_size, max_position_percentage, account_value, price):
    """Apply constraints to calculated position size.
    
    Args:
        position_size: Calculated position size
        max_position_size: Maximum position size in shares/contracts
        max_position_percentage: Maximum position size as percentage of account
        account_value: Total account value
        price: Current price of the asset
        
    Returns:
        Constrained position size
    """
    # Apply maximum position size constraint
    position_size = min(position_size, max_position_size)
    
    # Apply maximum position percentage constraint
    max_shares_by_pct = (account_value * max_position_percentage) / price
    position_size = min(position_size, max_shares_by_pct)
    
    # Round down to whole number of shares/contracts
    position_size = math.floor(position_size)
    
    return position_size
```

Key constraints:
- **Maximum Shares/Contracts**: Absolute limit on position size
- **Maximum Percentage**: Limit as percentage of account
- **Minimum Position Size**: Minimum viable position size
- **Lot Size Constraints**: Round to appropriate lot sizes

### 2. Liquidity Constraints

```python
# From src/trading_strategy/sizing/liquidity_manager.py
def apply_liquidity_constraints(position_size, symbol, avg_volume, price):
    """Adjust position size based on liquidity constraints.
    
    Args:
        position_size: Calculated position size
        symbol: Ticker symbol
        avg_volume: Average daily volume
        price: Current price of the asset
        
    Returns:
        Liquidity-adjusted position size
    """
    # Calculate dollar value of position
    position_value = position_size * price
    
    # Calculate average daily dollar volume
    avg_dollar_volume = avg_volume * price
    
    # Limit position to percentage of average daily volume
    max_pct_of_volume = 0.01  # 1% of average daily volume
    max_position_by_volume = (avg_volume * max_pct_of_volume)
    
    # Apply constraint
    adjusted_position_size = min(position_size, max_position_by_volume)
    
    return adjusted_position_size
```

Key liquidity factors:
- **Average Daily Volume**: Limits based on typical trading volume
- **Market Impact**: Estimated price impact of the trade
- **Spread Costs**: Wider spreads require smaller positions
- **Order Book Depth**: Available liquidity at different price levels

## Integration with Trading Strategy

The position sizing system integrates with the trading strategy subsystem:

```python
# From src/trading_strategy/execution/order_generator.py
def generate_order(signal, account_info, risk_params, market_data):
    """Generate an order based on a trading signal.
    
    Args:
        signal: Trading signal with symbol, direction, etc.
        account_info: Account information including balance
        risk_params: Risk parameters for position sizing
        market_data: Current market data for the symbol
        
    Returns:
        Order object with symbol, quantity, order type, etc.
    """
    # Extract signal information
    symbol = signal.symbol
    direction = signal.direction
    confidence = signal.confidence
    
    # Get current price and volatility
    current_price = market_data.get_last_price(symbol)
    atr = market_data.get_atr(symbol, timeframe="1d", period=14)
    
    # Calculate stop loss price
    stop_loss_price = calculate_stop_loss_price(current_price, direction, atr)
    
    # Calculate position size
    position_size = calculate_position_size_volatility_adjusted(
        account_value=account_info.balance,
        risk_per_trade=risk_params.risk_per_trade,
        atr=atr,
        atr_multiplier=risk_params.atr_multiplier
    )
    
    # Apply constraints
    position_size = apply_position_size_constraints(
        position_size=position_size,
        max_position_size=risk_params.max_position_size,
        max_position_percentage=risk_params.max_position_percentage,
        account_value=account_info.balance,
        price=current_price
    )
    
    # Apply liquidity constraints
    avg_volume = market_data.get_avg_volume(symbol, days=20)
    position_size = apply_liquidity_constraints(
        position_size=position_size,
        symbol=symbol,
        avg_volume=avg_volume,
        price=current_price
    )
    
    # Create order
    order = Order(
        symbol=symbol,
        quantity=position_size,
        order_type="market",
        side="buy" if direction == "long" else "sell",
        stop_loss_price=stop_loss_price
    )
    
    return order
```

## Conclusion

The risk management and position sizing system is a critical component of the Autonomous Trading System. It ensures that:

1. **Portfolio Risk is Controlled**: Through VaR limits, drawdown constraints, and exposure management
2. **Position Sizes are Optimized**: Based on risk parameters, market conditions, and strategy characteristics
3. **Losses are Limited**: Through stop losses, position sizing, and diversification
4. **Capital is Preserved**: By adapting to changing market conditions and limiting exposure

This multi-layered approach to risk management provides robust protection against excessive losses while allowing the system to capitalize on profitable opportunities.