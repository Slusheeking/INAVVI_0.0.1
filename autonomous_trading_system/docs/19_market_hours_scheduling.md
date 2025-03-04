# Market Hours and Scheduling

This document outlines how the Autonomous Trading System handles different market states, schedules operations based on market hours, and adapts to weekends and holidays.

## Market Calendar Management

The system uses a comprehensive market calendar to track trading hours, holidays, and special sessions:

```python
# From src/utils/time/market_calendar.py
class MarketCalendar:
    def __init__(self):
        # Load standard market calendars
        self.us_equity_calendar = self._load_us_equity_calendar()
        self.us_options_calendar = self._load_us_options_calendar()
        self.crypto_calendar = self._load_crypto_calendar()
        self.forex_calendar = self._load_forex_calendar()
        
        # Load holiday schedules
        self.holidays = self._load_holiday_schedule()
        
        # Load early close days
        self.early_close_days = self._load_early_close_schedule()
    
    def is_market_open(self, asset_class, timestamp=None):
        """Check if the market is open for a given asset class at a specific time."""
        timestamp = timestamp or datetime.now(timezone.utc)
        
        if asset_class == "us_equity":
            return self._is_us_equity_market_open(timestamp)
        elif asset_class == "us_options":
            return self._is_us_options_market_open(timestamp)
        elif asset_class == "crypto":
            return self._is_crypto_market_open(timestamp)  # Always True except for maintenance
        elif asset_class == "forex":
            return self._is_forex_market_open(timestamp)
        else:
            raise ValueError(f"Unknown asset class: {asset_class}")
    
    def get_next_market_open(self, asset_class, from_timestamp=None):
        """Get the next market open time for a given asset class."""
        # Implementation details...
    
    def get_next_market_close(self, asset_class, from_timestamp=None):
        """Get the next market close time for a given asset class."""
        # Implementation details...
    
    def is_holiday(self, date):
        """Check if a given date is a holiday."""
        return date in self.holidays
    
    def is_early_close(self, date):
        """Check if a given date has an early close."""
        return date in self.early_close_days
```

### Market Hours by Asset Class

The system handles different market hours for various asset classes:

| Asset Class | Regular Hours (ET) | Extended Hours (ET) | Weekend Trading | Holiday Schedule |
|-------------|-------------------|---------------------|----------------|------------------|
| US Equities | 9:30 AM - 4:00 PM | 4:00 AM - 9:30 AM, 4:00 PM - 8:00 PM | Closed | NYSE/NASDAQ Calendar |
| US Options | 9:30 AM - 4:00 PM | None | Closed | OCC Calendar |
| Cryptocurrencies | 24/7 | N/A | Open | Exchange Maintenance |
| Forex | Sunday 5:00 PM - Friday 5:00 PM | N/A | Partial | Major Bank Holidays |

### Holiday Calendar

The system maintains a comprehensive holiday calendar for all supported markets:

```python
# Example holiday calendar configuration
HOLIDAY_CALENDAR = {
    "us_equity": [
        {"date": "2025-01-01", "name": "New Year's Day", "status": "closed"},
        {"date": "2025-01-20", "name": "Martin Luther King Jr. Day", "status": "closed"},
        {"date": "2025-02-17", "name": "Presidents' Day", "status": "closed"},
        {"date": "2025-04-18", "name": "Good Friday", "status": "closed"},
        {"date": "2025-05-26", "name": "Memorial Day", "status": "closed"},
        {"date": "2025-06-19", "name": "Juneteenth", "status": "closed"},
        {"date": "2025-07-04", "name": "Independence Day", "status": "closed"},
        {"date": "2025-09-01", "name": "Labor Day", "status": "closed"},
        {"date": "2025-11-27", "name": "Thanksgiving Day", "status": "closed"},
        {"date": "2025-12-25", "name": "Christmas Day", "status": "closed"}
    ],
    "us_options": [
        # Similar to US equity with some differences
    ],
    "forex": [
        {"date": "2025-01-01", "name": "New Year's Day", "status": "limited"},
        {"date": "2025-04-18", "name": "Good Friday", "status": "closed"},
        {"date": "2025-04-21", "name": "Easter Monday", "status": "limited"},
        {"date": "2025-12-25", "name": "Christmas Day", "status": "closed"},
        {"date": "2025-12-26", "name": "Boxing Day", "status": "limited"}
    ]
}
```

## Data Availability by Market State

The system adapts to different data availability based on market state:

### During Regular Market Hours

During regular market hours, all data types are available:

| Data Type | US Equities | US Options | Crypto | Forex |
|-----------|------------|-----------|--------|-------|
| OHLCV Bars | Real-time | Real-time | Real-time | Real-time |
| Trades | Real-time | Real-time | Real-time | Real-time |
| Quotes | Real-time | Real-time | Real-time | Real-time |
| Order Book | Real-time | Real-time | Real-time | Real-time |
| Options Flow | Real-time | Real-time | N/A | N/A |

```python
# Example data acquisition during market hours
async def acquire_market_data(self, symbols, asset_class):
    """Acquire market data during regular hours."""
    if self.market_calendar.is_market_open(asset_class):
        # Use WebSocket for real-time data
        return await self.websocket_client.subscribe_to_data(symbols)
    else:
        # Use REST API for delayed/historical data
        return await self.rest_client.get_latest_data(symbols)
```

### During Extended Hours (Pre/Post Market)

During extended hours, data availability is more limited:

| Data Type | US Equities | US Options | Crypto | Forex |
|-----------|------------|-----------|--------|-------|
| OHLCV Bars | Limited | Not available | Real-time | Real-time |
| Trades | Limited | Not available | Real-time | Real-time |
| Quotes | Limited | Not available | Real-time | Real-time |
| Order Book | Limited | Not available | Real-time | Real-time |
| Options Flow | Not available | Not available | N/A | N/A |

```python
# Example handling of extended hours data
def process_extended_hours_data(self, data, asset_class):
    """Process data during extended hours."""
    if asset_class == "us_equity":
        # Apply extended hours flags and liquidity adjustments
        data["extended_hours"] = True
        data["liquidity_factor"] = 0.3  # Reduced liquidity
        
    return data
```

### During Weekends and Holidays

During weekends and holidays, data availability varies by asset class:

| Data Type | US Equities | US Options | Crypto | Forex |
|-----------|------------|-----------|--------|-------|
| OHLCV Bars | Historical only | Historical only | Real-time | Limited/Closed |
| Trades | Not available | Not available | Real-time | Limited/Closed |
| Quotes | Not available | Not available | Real-time | Limited/Closed |
| Order Book | Not available | Not available | Real-time | Limited/Closed |
| Options Flow | Not available | Not available | N/A | N/A |

```python
# Example weekend/holiday data handling
def get_weekend_data(self, symbols, asset_class):
    """Get data during weekends."""
    if asset_class == "crypto":
        # Crypto trades 24/7
        return self.get_realtime_data(symbols)
    elif asset_class in ["us_equity", "us_options"]:
        # Return latest historical data
        friday = self._get_previous_trading_day()
        return self.get_historical_data(symbols, friday)
```

## Scheduling System

The system uses a sophisticated scheduling system to coordinate operations based on market hours:

### Scheduler Architecture

```python
# From src/data_acquisition/pipeline/pipeline_scheduler.py
class PipelineScheduler:
    def __init__(self, market_calendar):
        self.market_calendar = market_calendar
        self.scheduler = AsyncIOScheduler()
        self.tasks = {}
    
    def schedule_market_dependent_task(self, task_func, asset_class, relation="during", **kwargs):
        """Schedule a task relative to market hours.
        
        Args:
            task_func: Function to execute
            asset_class: Asset class for market hours reference
            relation: Timing relation to market hours:
                - "during": Run during market hours
                - "before_open": Run before market open
                - "after_close": Run after market close
                - "weekend": Run on weekends
            **kwargs: Additional scheduling parameters
        """
        if relation == "during":
            self._schedule_during_market_hours(task_func, asset_class, **kwargs)
        elif relation == "before_open":
            self._schedule_before_market_open(task_func, asset_class, **kwargs)
        elif relation == "after_close":
            self._schedule_after_market_close(task_func, asset_class, **kwargs)
        elif relation == "weekend":
            self._schedule_weekend_task(task_func, asset_class, **kwargs)
    
    def _schedule_during_market_hours(self, task_func, asset_class, interval_minutes=5):
        """Schedule a task to run at intervals during market hours."""
        # Implementation details...
```

### Task Types by Market State

The system schedules different types of tasks based on market state:

#### During Market Hours Tasks

```python
# Example market hours task scheduling
scheduler.schedule_market_dependent_task(
    data_acquisition.collect_real_time_data,
    asset_class="us_equity",
    relation="during",
    interval_minutes=1
)

scheduler.schedule_market_dependent_task(
    trading_strategy.generate_signals,
    asset_class="us_equity",
    relation="during",
    interval_minutes=5
)

scheduler.schedule_market_dependent_task(
    risk_manager.update_positions,
    asset_class="us_equity",
    relation="during",
    interval_minutes=15
)
```

#### Pre-Market Tasks

```python
# Example pre-market task scheduling
scheduler.schedule_market_dependent_task(
    data_acquisition.prepare_daily_data,
    asset_class="us_equity",
    relation="before_open",
    minutes_before=60
)

scheduler.schedule_market_dependent_task(
    trading_strategy.prepare_daily_strategy,
    asset_class="us_equity",
    relation="before_open",
    minutes_before=30
)

scheduler.schedule_market_dependent_task(
    risk_manager.set_daily_risk_limits,
    asset_class="us_equity",
    relation="before_open",
    minutes_before=15
)
```

#### Post-Market Tasks

```python
# Example post-market task scheduling
scheduler.schedule_market_dependent_task(
    data_acquisition.finalize_daily_data,
    asset_class="us_equity",
    relation="after_close",
    minutes_after=15
)

scheduler.schedule_market_dependent_task(
    performance_analyzer.calculate_daily_metrics,
    asset_class="us_equity",
    relation="after_close",
    minutes_after=30
)

scheduler.schedule_market_dependent_task(
    model_training.update_models,
    asset_class="us_equity",
    relation="after_close",
    minutes_after=60
)
```

#### Weekend Tasks

```python
# Example weekend task scheduling
scheduler.schedule_market_dependent_task(
    data_acquisition.perform_data_maintenance,
    asset_class="us_equity",
    relation="weekend",
    day_of_week="Saturday",
    hour=10
)

scheduler.schedule_market_dependent_task(
    model_training.perform_full_retraining,
    asset_class="us_equity",
    relation="weekend",
    day_of_week="Sunday",
    hour=12
)
```

### Handling Market Disruptions

The system includes special handling for market disruptions:

```python
# Example market disruption handling
def handle_market_disruption(self, asset_class, disruption_type):
    """Handle unexpected market disruptions."""
    if disruption_type == "early_close":
        # Adjust schedules for early close
        self._reschedule_for_early_close(asset_class)
    elif disruption_type == "late_open":
        # Adjust schedules for late open
        self._reschedule_for_late_open(asset_class)
    elif disruption_type == "circuit_breaker":
        # Handle market-wide circuit breaker
        self._pause_trading_activities(asset_class)
        self._schedule_resumption(asset_class)
```

## Integration with Other Subsystems

The market hours and scheduling system integrates with other subsystems:

### Data Acquisition Integration

```python
# Example data acquisition integration
class DataAcquisitionManager:
    def __init__(self, market_calendar, scheduler):
        self.market_calendar = market_calendar
        self.scheduler = scheduler
        
    def setup_data_collection(self):
        """Set up data collection schedules based on market hours."""
        # Schedule real-time data collection during market hours
        self.scheduler.schedule_market_dependent_task(
            self.collect_real_time_data,
            asset_class="us_equity",
            relation="during",
            interval_minutes=1
        )
        
        # Schedule historical data updates after market close
        self.scheduler.schedule_market_dependent_task(
            self.update_historical_data,
            asset_class="us_equity",
            relation="after_close",
            minutes_after=30
        )
```

### Trading Strategy Integration

```python
# Example trading strategy integration
class TradingStrategyManager:
    def __init__(self, market_calendar, scheduler):
        self.market_calendar = market_calendar
        self.scheduler = scheduler
        
    def setup_strategy_execution(self):
        """Set up strategy execution schedules based on market hours."""
        # Only execute trades during market hours
        self.scheduler.schedule_market_dependent_task(
            self.execute_trading_signals,
            asset_class="us_equity",
            relation="during",
            interval_minutes=5,
            # Don't execute in first 15 minutes or last 5 minutes
            market_open_offset_minutes=15,
            market_close_offset_minutes=5
        )
```

### Monitoring Integration

```python
# Example monitoring integration
class MonitoringManager:
    def __init__(self, market_calendar, scheduler):
        self.market_calendar = market_calendar
        self.scheduler = scheduler
        
    def setup_monitoring(self):
        """Set up monitoring schedules."""
        # More frequent checks during market hours
        self.scheduler.schedule_market_dependent_task(
            self.check_system_health,
            asset_class="us_equity",
            relation="during",
            interval_minutes=5
        )
        
        # Less frequent checks outside market hours
        self.scheduler.schedule_market_dependent_task(
            self.check_system_health,
            asset_class="us_equity",
            relation="outside",
            interval_minutes=30
        )
```

## Conclusion

The market hours and scheduling system ensures that the Autonomous Trading System operates correctly across different market states, including regular hours, extended hours, weekends, and holidays. By adapting to the specific characteristics of each asset class and market state, the system optimizes data collection, trading activity, and system maintenance.

Key benefits of this approach include:
- Efficient resource utilization by focusing on active markets
- Proper handling of different data availability scenarios
- Coordinated scheduling of interdependent tasks
- Automated adaptation to market calendars and special events
- Resilience to market disruptions and unexpected changes