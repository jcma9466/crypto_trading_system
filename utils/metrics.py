import numpy as np
import pandas as pd
import quantstats as qs

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: cumulative returns
    """
    # Convert numpy array to pandas Series if needed
    if isinstance(returns_pct, np.ndarray):
        # Create a simple datetime index for quantstats compatibility
        date_range = pd.date_range(start='2020-01-01', periods=len(returns_pct), freq='D')
        returns_pct = pd.Series(returns_pct, index=date_range)
    return qs.stats.comp(returns_pct)

def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t
    risk_free: risk free rate

    return: float
    """
    # Convert numpy array to pandas Series if needed
    if isinstance(returns_pct, np.ndarray):
        # Create a simple datetime index for quantstats compatibility
        date_range = pd.date_range(start='2020-01-01', periods=len(returns_pct), freq='D')
        returns_pct = pd.Series(returns_pct, index=date_range)
    return qs.stats.sharpe(returns_pct, rf=risk_free)

def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    # Convert numpy array to pandas Series if needed
    if isinstance(returns_pct, np.ndarray):
        # Create a simple datetime index for quantstats compatibility
        date_range = pd.date_range(start='2020-01-01', periods=len(returns_pct), freq='D')
        returns_pct = pd.Series(returns_pct, index=date_range)
    return qs.stats.max_drawdown(returns_pct)

def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    cum_returns = cumulative_returns(returns_pct)
    # Handle both scalar and series returns from cumulative_returns
    if hasattr(cum_returns, 'iloc'):
        returns = cum_returns.iloc[-1]  # Get last value if it's a Series
    else:
        returns = cum_returns  # Use directly if it's a scalar
    if mdd == 0:
        return np.inf
    return returns/mdd


