import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Define economic scenarios with their impact parameters
ECONOMIC_SCENARIOS = {
    "Normal Market": {
        "description": "Base case scenario with normal market conditions",
        "returns_adjustment": 0.0,  # No adjustment
        "volatility_adjustment": 1.0,  # No adjustment
        "correlation_adjustment": 0.0,  # No adjustment
        "drawdown_adjustment": 0.0,  # No adjustment
        "impact_factor": {
            "Technology": 0.0,
            "Financial": 0.0,
            "Healthcare": 0.0,
            "Energy": 0.0,
            "Consumer": 0.0,
            "Industrial": 0.0,
            "Materials": 0.0,
            "Utilities": 0.0,
            "Real Estate": 0.0
        }
    },
    "Market Crash": {
        "description": "Severe and sudden market downturn across all sectors",
        "returns_adjustment": -0.25,  # Significant negative adjustment to returns
        "volatility_adjustment": 2.5,  # Increase volatility by 2.5x
        "correlation_adjustment": 0.3,  # Increased correlations during crisis
        "drawdown_adjustment": 0.15,  # Additional drawdown
        "impact_factor": {
            "Technology": -0.3,
            "Financial": -0.35,
            "Healthcare": -0.2,
            "Energy": -0.25,
            "Consumer": -0.25,
            "Industrial": -0.3,
            "Materials": -0.25,
            "Utilities": -0.15,
            "Real Estate": -0.3
        }
    },
    "Recession": {
        "description": "Economic contraction with prolonged negative growth",
        "returns_adjustment": -0.15,  # Negative adjustment to returns
        "volatility_adjustment": 1.8,  # Increase volatility
        "correlation_adjustment": 0.2,  # Increased correlations
        "drawdown_adjustment": 0.1,  # Additional drawdown
        "impact_factor": {
            "Technology": -0.2,
            "Financial": -0.25,
            "Healthcare": -0.1,
            "Energy": -0.2,
            "Consumer": -0.2,
            "Industrial": -0.25,
            "Materials": -0.2,
            "Utilities": -0.1,
            "Real Estate": -0.25
        }
    },
    "Inflation Surge": {
        "description": "Rapid increase in inflation rates affecting purchasing power",
        "returns_adjustment": -0.05,  # Modest negative adjustment to returns
        "volatility_adjustment": 1.4,  # Increase volatility
        "correlation_adjustment": 0.1,  # Modest increase in correlations
        "drawdown_adjustment": 0.05,  # Modest additional drawdown
        "impact_factor": {
            "Technology": -0.15,
            "Financial": 0.05,  # Banks might benefit from higher rates
            "Healthcare": -0.1,
            "Energy": 0.1,  # Energy often benefits from inflation
            "Consumer": -0.2,  # Consumer spending decreases
            "Industrial": -0.1,
            "Materials": 0.05,  # Materials can pass on costs
            "Utilities": -0.05,
            "Real Estate": -0.2  # Real estate sensitive to rates
        }
    },
    "Tech Bubble Burst": {
        "description": "Sharp correction in technology sector valuations",
        "returns_adjustment": -0.1,  # Negative adjustment to returns
        "volatility_adjustment": 1.6,  # Increase volatility
        "correlation_adjustment": 0.1,  # Modest increase in correlations
        "drawdown_adjustment": 0.08,  # Additional drawdown
        "impact_factor": {
            "Technology": -0.4,  # Tech heavily impacted
            "Financial": -0.15,
            "Healthcare": -0.05,
            "Energy": 0.0,
            "Consumer": -0.1,
            "Industrial": -0.1,
            "Materials": -0.05,
            "Utilities": 0.05,  # Defensive sectors might benefit
            "Real Estate": -0.1
        }
    },
    "Pandemic": {
        "description": "Global health crisis affecting economic activity",
        "returns_adjustment": -0.2,  # Significant negative adjustment to returns
        "volatility_adjustment": 2.2,  # Increase volatility significantly
        "correlation_adjustment": 0.25,  # Increased correlations
        "drawdown_adjustment": 0.12,  # Additional drawdown
        "impact_factor": {
            "Technology": 0.1,  # Tech might benefit from stay-at-home trends
            "Financial": -0.25,
            "Healthcare": 0.15,  # Healthcare might benefit
            "Energy": -0.3,  # Energy hit by reduced transportation
            "Consumer": -0.2,  # Split impact (discretionary vs. staples)
            "Industrial": -0.25,
            "Materials": -0.2,
            "Utilities": -0.1,
            "Real Estate": -0.25  # Commercial real estate impacted
        }
    }
}

# Sector mapping for common stocks
SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'AMZN': 'Technology', 'NVDA': 'Technology', 'ADBE': 'Technology',
    'CRM': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology', 'AMD': 'Technology',
    
    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'C': 'Financial',
    'GS': 'Financial', 'MS': 'Financial', 'BLK': 'Financial', 'AXP': 'Financial',
    'V': 'Financial', 'MA': 'Financial', 'BRK-A': 'Financial', 'BRK-B': 'Financial',
    
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare', 'ABT': 'Healthcare',
    'UNH': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare', 'LLY': 'Healthcare',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
    'SLB': 'Energy', 'BP': 'Energy', 'RDS-A': 'Energy', 'TOT': 'Energy',
    
    # Consumer
    'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
    'COST': 'Consumer', 'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',
    
    # Industrial
    'GE': 'Industrial', 'HON': 'Industrial', 'MMM': 'Industrial', 'BA': 'Industrial',
    'CAT': 'Industrial', 'DE': 'Industrial', 'UPS': 'Industrial', 'FDX': 'Industrial',
    
    # Indonesian Stocks
    # Banking
    'BBRI.JK': 'Financial', 'BBCA.JK': 'Financial', 'BMRI.JK': 'Financial', 'BBNI.JK': 'Financial', 
    'BRIS.JK': 'Financial', 'BJTM.JK': 'Financial', 'BTPS.JK': 'Financial',
    
    # Consumer Goods
    'UNVR.JK': 'Consumer', 'ICBP.JK': 'Consumer', 'INDF.JK': 'Consumer', 'KLBF.JK': 'Consumer',
    'SIDO.JK': 'Consumer', 'MYOR.JK': 'Consumer', 'GGRM.JK': 'Consumer', 'HMSP.JK': 'Consumer',
    
    # Telecommunications
    'TLKM.JK': 'Technology', 'ISAT.JK': 'Technology', 'EXCL.JK': 'Technology',
    
    # Energy/Mining
    'ADRO.JK': 'Energy', 'PTBA.JK': 'Energy', 'ITMG.JK': 'Energy', 'MEDC.JK': 'Energy',
    'INCO.JK': 'Materials', 'ANTM.JK': 'Materials', 'TINS.JK': 'Materials',
    
    # Property/Infrastructure
    'SMGR.JK': 'Industrial', 'WIKA.JK': 'Industrial', 'WSKT.JK': 'Industrial', 'PTPP.JK': 'Industrial',
    'BSDE.JK': 'Real Estate', 'CTRA.JK': 'Real Estate', 'PWON.JK': 'Real Estate',
    
    # Default sector for unknown symbols
    'DEFAULT': 'Unknown'
}

def get_symbol_sector(symbol: str) -> str:
    """
    Get the sector for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        str: Sector classification
    """
    return SECTOR_MAPPING.get(symbol, SECTOR_MAPPING['DEFAULT'])

def apply_economic_scenario(
    historical_data: pd.DataFrame,
    scenario_params: Dict
) -> pd.DataFrame:
    """
    Apply economic scenario adjustments to historical price data.
    
    Args:
        historical_data: DataFrame with historical asset prices
        scenario_params: Parameters for the economic scenario
        
    Returns:
        DataFrame: Adjusted historical data
    """
    # Create a copy of the data to avoid modifying the original
    adjusted_data = historical_data.copy()
    
    # Extract adjustment parameters
    returns_adj = scenario_params.get('returns_adjustment', 0.0)
    vol_adj = scenario_params.get('volatility_adjustment', 1.0)
    corr_adj = scenario_params.get('correlation_adjustment', 0.0)
    impact_factors = scenario_params.get('impact_factor', {})
    
    # Calculate returns
    returns = historical_data.pct_change().dropna()
    
    # Apply sector-specific adjustments
    for column in returns.columns:
        # Get sector
        sector = get_symbol_sector(column)
        
        # Get sector-specific impact
        sector_impact = impact_factors.get(sector, 0.0)
        
        # Combine global and sector-specific adjustments
        total_returns_adj = returns_adj + sector_impact
        
        # Adjust returns
        adjusted_returns = returns[column] * vol_adj + total_returns_adj / 252  # Daily adjustment
        
        # Replace returns in the adjusted DataFrame
        start_idx = returns.index[0]
        start_price = adjusted_data.loc[start_idx, column]
        
        # Reconstruct prices from adjusted returns
        for i, date in enumerate(returns.index):
            if i == 0:
                adjusted_data.loc[date, column] = start_price
            else:
                prev_date = returns.index[i-1]
                prev_price = adjusted_data.loc[prev_date, column]
                adjusted_data.loc[date, column] = prev_price * (1 + adjusted_returns[date])
    
    return adjusted_data

def generate_custom_scenario(
    base_scenario: str,
    custom_adjustments: Dict = None
) -> Dict:
    """
    Generate a custom economic scenario based on a predefined scenario with custom adjustments.
    
    Args:
        base_scenario: Name of the base scenario to customize
        custom_adjustments: Dict with custom adjustment values
        
    Returns:
        Dict: Custom scenario parameters
    """
    if base_scenario not in ECONOMIC_SCENARIOS:
        raise ValueError(f"Base scenario '{base_scenario}' not found")
    
    # Create copy of base scenario
    custom_scenario = ECONOMIC_SCENARIOS[base_scenario].copy()
    
    # If no custom adjustments provided, return the base scenario
    if not custom_adjustments:
        return custom_scenario
    
    # Apply custom adjustments
    for key, value in custom_adjustments.items():
        if key in custom_scenario:
            if isinstance(custom_scenario[key], dict) and isinstance(value, dict):
                # For nested dictionaries like impact_factor
                for subkey, subvalue in value.items():
                    if subkey in custom_scenario[key]:
                        custom_scenario[key][subkey] = subvalue
            else:
                # For direct parameters
                custom_scenario[key] = value
    
    return custom_scenario

def get_scenario_description(scenario_name: str) -> str:
    """
    Get the description for a given economic scenario.
    
    Args:
        scenario_name: Name of the economic scenario
        
    Returns:
        str: Description of the scenario
    """
    scenario = ECONOMIC_SCENARIOS.get(scenario_name)
    if scenario:
        return scenario.get('description', 'No description available')
    else:
        return "Scenario not found"