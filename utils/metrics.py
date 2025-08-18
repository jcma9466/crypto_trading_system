import numpy as np
import pandas as pd
import quantstats as qs

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    return qs.stats.comp(returns_pct)

def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t
    risk_free: risk free rate

    return: float
    """
    return qs.stats.sharpe(returns_pct, rf=risk_free)

def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    return qs.stats.max_drawdown(returns_pct)

def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct)[len(returns_pct)-1]
    if mdd == 0:
        return np.inf
    return returns/mdd


