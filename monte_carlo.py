import numpy as np
import pandas as pd
from typing import Dict, List, Any

def run_monte_carlo_simulation(
    historical_data: pd.DataFrame,
    portfolio_data: pd.DataFrame,
    num_simulations: int = 1000,
    time_horizon: int = 21,
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for portfolio performance.
    
    Args:
        historical_data: DataFrame with historical asset prices
        portfolio_data: DataFrame with portfolio weights and symbols
        num_simulations: Number of simulations to run
        time_horizon: Time horizon in trading days
        random_seed: Seed for random number generator
        
    Returns:
        Dict: Simulation results including paths and metrics
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate daily returns
    returns = historical_data.pct_change().dropna()
    
    # Extract portfolio information
    symbols = portfolio_data['Symbol'].tolist()
    
    # Filter returns to include only portfolio assets
    portfolio_returns = returns[[col for col in returns.columns if col in symbols]]
    
    # Handle missing columns - some symbols might not have data
    missing_symbols = [symbol for symbol in symbols if symbol not in portfolio_returns.columns]
    if missing_symbols:
        print(f"Warning: No historical data found for: {', '.join(missing_symbols)}")
    
    # Create weights array matching the returns columns
    weights = np.zeros(len(portfolio_returns.columns))
    for i, symbol in enumerate(portfolio_returns.columns):
        symbol_weight = portfolio_data.loc[portfolio_data['Symbol'] == symbol, 'Weight'].values
        if len(symbol_weight) > 0:
            weights[i] = symbol_weight[0]
    
    # Normalize weights to sum to 1
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        raise ValueError("No valid weights could be calculated")
    
    # Calculate portfolio statistics
    mean_daily_return = np.sum(portfolio_returns.mean() * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns.cov(), weights)))
    
    # Store simulation paths
    simulations = []
    final_returns = []
    max_drawdowns = []
    
    # Run simulations
    for _ in range(num_simulations):
        # Generate random returns
        random_returns = np.random.normal(
            mean_daily_return, 
            portfolio_volatility, 
            time_horizon
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + random_returns) - 1
        
        # Calculate maximum drawdown
        peak = 0
        max_drawdown = 0
        
        for return_value in cumulative_returns:
            if return_value > peak:
                peak = return_value
            drawdown = (peak - return_value) / (1 + peak)  # Percentage drawdown
            max_drawdown = max(max_drawdown, drawdown)
        
        # Store results
        simulations.append(cumulative_returns)
        final_returns.append(cumulative_returns[-1])
        max_drawdowns.append(-max_drawdown)  # Store as negative value
    
    # Convert lists to numpy arrays
    simulations = np.array(simulations)
    final_returns = np.array(final_returns)
    max_drawdowns = np.array(max_drawdowns)
    
    # Calculate percentiles
    percentiles = {}
    for percentile in [5, 25, 50, 75, 95]:
        percentiles[percentile] = np.percentile(simulations, percentile, axis=0)
    
    # Return simulation results
    return {
        'simulations': simulations,
        'final_returns': final_returns,
        'max_drawdowns': max_drawdowns,
        'percentiles': percentiles,
        'time_horizon': time_horizon,
        'mean_return': mean_daily_return,
        'volatility': portfolio_volatility,
        'weights': weights,
        'assets': portfolio_returns.columns.tolist()
    }

def calculate_portfolio_metrics(simulations: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various portfolio performance metrics from simulation results.
    
    Args:
        simulations: Dict containing simulation data
        
    Returns:
        Dict: Portfolio performance metrics
    """
    # Extract final returns
    final_returns = simulations['final_returns']
    
    # Calculate metrics
    mean_return = np.mean(final_returns)
    median_return = np.median(final_returns)
    std_return = np.std(final_returns)
    
    # Calculate percentiles
    percentile_5 = np.percentile(final_returns, 5)
    percentile_95 = np.percentile(final_returns, 95)
    
    # Return metrics
    return {
        'mean_return': mean_return,
        'median_return': median_return,
        'std_return': std_return,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95
    }