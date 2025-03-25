import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import statsmodels.api as sm
import pandas.tseries.offsets as offsets

def run_arima_forecast(
    historical_data: pd.DataFrame,
    portfolio_data: pd.DataFrame,
    forecast_periods: int = 21,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Run ARIMA time series forecast for portfolio.
    
    Args:
        historical_data: DataFrame with historical asset prices
        portfolio_data: DataFrame with portfolio weights and symbols
        forecast_periods: Number of periods to forecast
        confidence_level: Confidence level for prediction intervals
        
    Returns:
        Dict: Forecast results
    """
    # Calculate historical portfolio values
    symbols = portfolio_data['Symbol'].tolist()
    weights = portfolio_data.set_index('Symbol')['Weight']
    
    # Filter to keep only symbols that exist in historical data
    available_symbols = [symbol for symbol in symbols if symbol in historical_data.columns]
    if not available_symbols:
        raise ValueError("None of the portfolio assets have historical data")
    
    # Normalize weights if some assets are missing
    if len(available_symbols) < len(symbols):
        weights = weights[available_symbols]
        weights = weights / weights.sum()
    
    # Calculate weighted portfolio values (normalize to 100 at the start)
    historical_data_filtered = historical_data[available_symbols].copy()
    # Handle NaN values by forward filling, then backward filling any remaining NaNs
    historical_data_filtered = historical_data_filtered.ffill().bfill()
    
    # Calculate the weighted sum of asset prices
    portfolio_values_raw = historical_data_filtered.dot(weights)
    
    # Create a normalized version starting at 100
    portfolio_values = 100 * portfolio_values_raw / portfolio_values_raw.iloc[0]
    
    # Ensure the index has frequency information for ARIMA
    if not isinstance(portfolio_values.index, pd.DatetimeIndex):
        print("Warning: Index is not a DatetimeIndex, converting...")
        try:
            portfolio_values.index = pd.DatetimeIndex(portfolio_values.index)
        except Exception as e:
            print(f"Could not convert index to DatetimeIndex: {str(e)}")
    
    # Set a frequency if not present
    if portfolio_values.index.freq is None:
        print("Setting daily frequency to time series index")
        # Create a new series with a proper frequency
        dates = pd.date_range(start=portfolio_values.index[0], 
                             end=portfolio_values.index[-1], 
                             freq='B')  # 'B' for business days
        
        # Reindex the series to the new date range
        portfolio_values = portfolio_values.reindex(dates).interpolate(method='linear')
    
    # Fit ARIMA model
    try:
        # Try auto ARIMA
        model = sm.tsa.ARIMA(
            portfolio_values, 
            order=(2, 1, 2),  # Default p=2, d=1, q=2
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit()
        
        # Create forecast with proper date index
        last_date = portfolio_values.index[-1]
        forecast_index = pd.date_range(start=last_date + offsets.BDay(1), 
                                      periods=forecast_periods, 
                                      freq='B')
        
        # Get forecast
        forecast = result.get_forecast(steps=forecast_periods)
        forecast_values = forecast.predicted_mean
        forecast_values.index = forecast_index
        
        # Get confidence intervals
        prediction_intervals = forecast.conf_int(alpha=(1 - confidence_level))
        prediction_intervals.index = forecast_index
        lower_ci = prediction_intervals.iloc[:, 0]
        upper_ci = prediction_intervals.iloc[:, 1]
        
        # Return results
        return {
            'historical_values': portfolio_values,
            'forecast_values': forecast_values,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'model_summary': result.summary()
        }
    
    except Exception as e:
        print(f"ARIMA model fitting error: {str(e)}")
        
        # Fallback to simple moving average if ARIMA fails
        return run_simple_forecast(
            portfolio_values, 
            forecast_periods, 
            confidence_level
        )

def run_simple_forecast(
    series: pd.Series,
    forecast_periods: int = 21,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Run a simple forecast based on moving average.
    
    Args:
        series: Time series data
        forecast_periods: Number of periods to forecast
        confidence_level: Confidence level for prediction intervals
        
    Returns:
        Dict: Forecast results
    """
    # Calculate mean and standard deviation of returns
    returns = series.pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Calculate Z-score for confidence interval
    z_score = abs(np.quantile(np.random.normal(0, 1, 10000), (1 - confidence_level) / 2))
    
    # Create forecast
    last_value = series.iloc[-1]
    
    # Use business day frequency for forecast (consistent with ARIMA)
    last_date = series.index[-1]
    forecast_index = pd.date_range(
        start=last_date + offsets.BDay(1), 
        periods=forecast_periods, 
        freq='B'
    )
    
    # Create forecast with exponential growth model
    forecast_values = pd.Series(
        index=forecast_index,
        data=[last_value * (1 + mean_return) ** (i + 1) for i in range(forecast_periods)]
    )
    
    # Create confidence intervals
    lower_ci = pd.Series(
        index=forecast_index,
        data=[last_value * (1 + mean_return - z_score * std_return) ** (i + 1) for i in range(forecast_periods)]
    )
    
    upper_ci = pd.Series(
        index=forecast_index,
        data=[last_value * (1 + mean_return + z_score * std_return) ** (i + 1) for i in range(forecast_periods)]
    )
    
    return {
        'historical_values': series,
        'forecast_values': forecast_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'model_summary': "Simple exponential forecast (ARIMA failed)"
    }

def calculate_var(
    simulation_results: Dict,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        confidence_level: Confidence level for VaR calculation (default: 0.95)
        
    Returns:
        float: Value at Risk as a percentage of portfolio value
    """
    # Extract final returns from simulations
    final_returns = simulation_results['final_returns']
    
    # Calculate VaR
    var_percentile = 100 * (1 - confidence_level)
    var = -np.percentile(final_returns, var_percentile)
    
    return var

def calculate_expected_shortfall(
    simulation_results: Dict,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR) from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        confidence_level: Confidence level for ES calculation (default: 0.95)
        
    Returns:
        float: Expected Shortfall as a percentage of portfolio value
    """
    # Extract final returns from simulations
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

def calculate_conditional_drawdown(
    simulation_results: Dict,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Drawdown at Risk (CDaR) from simulation results.
    
    Args:
        simulation_results: Dict containing simulation data
        confidence_level: Confidence level for CDaR calculation (default: 0.95)
        
    Returns:
        float: Conditional Drawdown at Risk
    """
    # Extract max drawdowns from simulations
    max_drawdowns = np.array(simulation_results['max_drawdowns'])
    
    # Calculate Drawdown at Risk (DaR)
    dar_percentile = 100 * (1 - confidence_level)
    dar = -np.percentile(max_drawdowns, dar_percentile)
    
    # Calculate Conditional Drawdown at Risk (CDaR)
    conditional_drawdowns = -max_drawdowns[max_drawdowns < -dar]
    
    if len(conditional_drawdowns) == 0:
        return -dar  # Fallback if no drawdowns exceed DaR
        
    cdar = np.mean(conditional_drawdowns)
    
    return cdar

def calculate_portfolio_metrics(
    simulation_results: Dict
) -> Dict:
    """
    Calculate comprehensive portfolio risk metrics.
    
    Args:
        simulation_results: Dict containing simulation data
        
    Returns:
        Dict: Various portfolio risk metrics
    """
    # Extract simulation data
    final_returns = simulation_results['final_returns']
    max_drawdowns = np.array(simulation_results['max_drawdowns'])
    
    # Calculate basic statistics
    mean_return = np.mean(final_returns)
    std_return = np.std(final_returns)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    
    # Calculate risk metrics
    var_95 = calculate_var(simulation_results, 0.95)
    var_99 = calculate_var(simulation_results, 0.99)
    es_95 = calculate_expected_shortfall(simulation_results, 0.95)
    es_99 = calculate_expected_shortfall(simulation_results, 0.99)
    cdar_95 = calculate_conditional_drawdown(simulation_results, 0.95)
    
    # Calculate additional metrics
    sortino_ratio = mean_return / np.std(final_returns[final_returns < 0]) if np.any(final_returns < 0) else float('inf')
    avg_drawdown = np.mean(-max_drawdowns)
    max_drawdown = np.min(max_drawdowns)
    
    # Compile results
    metrics = {
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'VaR_95': var_95,
        'VaR_99': var_99,
        'ES_95': es_95,
        'ES_99': es_99,
        'avg_drawdown': avg_drawdown,
        'max_drawdown': max_drawdown,
        'cdar_95': cdar_95
    }
    
    return metrics