import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union

def calculate_var(
    simulation_results: Dict[str, Any],
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR) from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        confidence_level: Confidence level for VaR (default: 0.95)
        method: Method to calculate VaR ('historical', 'parametric', 'cornish_fisher')
        
    Returns:
        float: Value at Risk as a percentage
    """
    if method not in ['historical', 'parametric', 'cornish_fisher']:
        raise ValueError(f"Invalid method: {method}. Choose 'historical', 'parametric', or 'cornish_fisher'")
    
    # Extract final returns
    final_returns = simulation_results['final_returns']
    
    if method == 'historical':
        # Historical VaR calculation
        percentile = 100 * (1 - confidence_level)
        var = -np.percentile(final_returns, percentile)
        
    elif method == 'parametric':
        # Parametric VaR calculation
        mean = np.mean(final_returns)
        std = np.std(final_returns)
        z_score = -np.quantile(np.random.normal(0, 1, 10000), 1 - confidence_level)
        var = -(mean + z_score * std)
        
    elif method == 'cornish_fisher':
        # Cornish-Fisher VaR calculation (adjusts for skew and kurtosis)
        mean = np.mean(final_returns)
        std = np.std(final_returns)
        skew = float(pd.Series(final_returns).skew())
        kurt = float(pd.Series(final_returns).kurtosis())
        
        z_score = -np.quantile(np.random.normal(0, 1, 10000), 1 - confidence_level)
        z_cf = (z_score + 
                (z_score**2 - 1) * skew / 6 + 
                (z_score**3 - 3 * z_score) * kurt / 24 - 
                (2 * z_score**3 - 5 * z_score) * skew**2 / 36)
        
        var = -(mean + z_cf * std)
    
    return var

def calculate_expected_shortfall(
    simulation_results: Dict[str, Any],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (ES) or Conditional VaR from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        confidence_level: Confidence level for ES (default: 0.95)
        
    Returns:
        float: Expected Shortfall as a percentage
    """
    # Extract final returns
    final_returns = simulation_results['final_returns']
    
    # Calculate VaR
    var_percentile = 100 * (1 - confidence_level)
    var = -np.percentile(final_returns, var_percentile)
    
    # Calculate Expected Shortfall
    losses = -final_returns
    conditional_losses = losses[losses > var]
    
    if len(conditional_losses) == 0:
        return var  # Fallback if no values exceed VaR
        
    expected_shortfall = np.mean(conditional_losses)
    
    return expected_shortfall

def calculate_drawdown_metrics(
    simulation_results: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate drawdown-based risk metrics from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Dict: Drawdown risk metrics
    """
    # Extract max drawdowns
    max_drawdowns = np.array(simulation_results['max_drawdowns'])
    
    # Calculate average drawdown
    avg_drawdown = np.mean(-max_drawdowns)
    
    # Calculate max drawdown (worst case)
    max_dd = np.min(max_drawdowns)
    
    # Calculate Conditional Drawdown at Risk (CDaR)
    # Find the threshold drawdown at the specified confidence level
    dar_percentile = 100 * (1 - confidence_level)
    dar = -np.percentile(max_drawdowns, dar_percentile)
    
    # Calculate the average of drawdowns exceeding the threshold
    conditional_drawdowns = -max_drawdowns[max_drawdowns < -dar]
    
    if len(conditional_drawdowns) == 0:
        cdar = -dar  # Fallback if no drawdowns exceed the threshold
    else:
        cdar = np.mean(conditional_drawdowns)
    
    return {
        'avg_drawdown': avg_drawdown,
        'max_drawdown': -max_dd,  # Convert to positive value
        'drawdown_at_risk': dar,
        'conditional_drawdown_at_risk': cdar
    }

def calculate_risk_adjusted_metrics(
    simulation_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate risk-adjusted performance metrics from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        
    Returns:
        Dict: Risk-adjusted performance metrics
    """
    # Extract simulation data
    final_returns = simulation_results['final_returns']
    annualization_factor = np.sqrt(252 / simulation_results['time_horizon'])  # Annualize based on trading days
    
    # Calculate mean return and volatility
    mean_return = np.mean(final_returns) * annualization_factor
    volatility = np.std(final_returns) * annualization_factor
    
    # Calculate downside deviation (returns below 0)
    negative_returns = final_returns[final_returns < 0]
    downside_deviation = np.std(negative_returns) * annualization_factor if len(negative_returns) > 0 else 1e-6
    
    # Calculate risk-free rate (assume 0 for simplicity)
    risk_free_rate = 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate Sortino ratio
    sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate Calmar ratio
    drawdown_metrics = calculate_drawdown_metrics(simulation_results)
    max_drawdown = drawdown_metrics['max_drawdown']
    calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'annualized_return': mean_return,
        'annualized_volatility': volatility,
        'downside_deviation': downside_deviation
    }

def calculate_comprehensive_risk_profile(
    simulation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate a comprehensive risk profile from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        
    Returns:
        Dict: Comprehensive risk metrics
    """
    # Calculate VaR and ES at different confidence levels
    var_90 = calculate_var(simulation_results, 0.90)
    var_95 = calculate_var(simulation_results, 0.95)
    var_99 = calculate_var(simulation_results, 0.99)
    
    es_90 = calculate_expected_shortfall(simulation_results, 0.90)
    es_95 = calculate_expected_shortfall(simulation_results, 0.95)
    es_99 = calculate_expected_shortfall(simulation_results, 0.99)
    
    # Calculate drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(simulation_results)
    
    # Calculate risk-adjusted metrics
    risk_adjusted_metrics = calculate_risk_adjusted_metrics(simulation_results)
    
    # Combine all metrics
    risk_profile = {
        'var': {
            '90%': var_90,
            '95%': var_95,
            '99%': var_99
        },
        'expected_shortfall': {
            '90%': es_90,
            '95%': es_95,
            '99%': es_99
        },
        'drawdown_metrics': drawdown_metrics,
        'risk_adjusted_metrics': risk_adjusted_metrics
    }
    
    return risk_profile