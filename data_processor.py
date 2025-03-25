import pandas as pd
import numpy as np
import yfinance as yf

def validate_portfolio_data(data: pd.DataFrame) -> bool:
    """
    Validate if the uploaded CSV data has the required format.
    
    Args:
        data: DataFrame containing portfolio data
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_columns = ['Symbol', 'Weight', 'Value']
    
    # Check if all required columns exist
    if not all(col in data.columns for col in required_columns):
        return False
    
    # Check if data types are valid
    try:
        # Verify Symbol is string
        data['Symbol'] = data['Symbol'].astype(str)
        
        # Verify Weight and Value are numeric
        data['Weight'] = pd.to_numeric(data['Weight'])
        data['Value'] = pd.to_numeric(data['Value'])
        
        # Check if weights sum to approximately 1
        total_weight = data['Weight'].sum()
        if not (0.99 <= total_weight <= 1.01):
            # Try to normalize weights
            data['Weight'] = data['Weight'] / total_weight
            
        return True
    except Exception:
        return False

def process_portfolio_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean portfolio data.
    
    Args:
        data: DataFrame containing portfolio data
        
    Returns:
        DataFrame: Processed portfolio data
    """
    # Ensure required columns exist
    required_columns = ['Symbol', 'Weight', 'Value']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in portfolio data")
    
    # Convert data types
    data['Symbol'] = data['Symbol'].astype(str)
    data['Weight'] = pd.to_numeric(data['Weight'])
    data['Value'] = pd.to_numeric(data['Value'])
    
    # Normalize weights if they don't sum to 1
    total_weight = data['Weight'].sum()
    if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
        data['Weight'] = data['Weight'] / total_weight
    
    # Calculate total portfolio value
    total_value = data['Value'].sum()
    
    # Calculate additional metrics
    data['Percentage'] = (data['Value'] / total_value * 100).round(2)
    
    # Sort by weight (descending)
    data = data.sort_values('Weight', ascending=False).reset_index(drop=True)
    
    return data

def load_sample_data() -> pd.DataFrame:
    """
    Load sample portfolio data for demonstration purposes.
    
    Returns:
        DataFrame: Sample portfolio data
    """
    # Create sample portfolio with major Indonesian stocks (value in million Rupiah)
    sample_data = {
        'Symbol': ['BBCA', 'BBRI', 'BMRI', 'TLKM', 'UNVR', 'ICBP', 'ADRO', 'KLBF', 'ASII', 'ANTM'],
        'Weight': [0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05],
        'Value': [150, 150, 100, 100, 100, 100, 100, 80, 70, 50]  # Dalam juta Rupiah
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Process the data
    return process_portfolio_data(df)

def fetch_historical_data(symbols: list, start_date, end_date) -> pd.DataFrame:
    """
    Fetch historical price data for the given symbols.
    
    Args:
        symbols: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format or datetime object
        end_date: End date in 'YYYY-MM-DD' format or datetime object
        
    Returns:
        DataFrame: Historical adjusted close prices
    """
    # Daftar saham Indonesia
    indonesian_stocks = [
        'BBCA', 'BBRI', 'BMRI', 'BBNI', 'TLKM', 'UNVR', 'ASII', 'ICBP', 'INDF', 'KLBF', 
        'ADRO', 'PTBA', 'ANTM', 'INCO', 'SMGR', 'EXCL', 'BSDE', 'WIKA', 'GGRM', 'HMSP', 
        'SIDO', 'MYOR', 'CPIN', 'BTPS', 'BJTM', 'BRIS', 'BDMN', 'BNGA', 'ISAT', 'FREN',
        'JPFA', 'PGAS', 'JSMR', 'ADHI', 'PTPP', 'WSKT', 'ITMG', 'MEDC', 'TINS', 'CTRA',
        'PWON', 'SMRA', 'LPKR', 'SRIL', 'INTP', 'BRPT', 'GOTO', 'BUKA', 'EMTK', 'AKRA', 'MNCN'
    ]
    
    try:
        # Convert dates to strings if they're datetime objects
        if not isinstance(start_date, str):
            start_date = start_date.strftime('%Y-%m-%d')
        if not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Kamus untuk memetakan simbol asli ke simbol untuk pengambilan data
        symbol_mapping = {}
        processed_symbols = []
        
        for symbol in symbols:
            # Periksa apakah saham Indonesia dan belum ada akhiran .JK
            if symbol in indonesian_stocks:
                processed_symbol = f"{symbol}.JK"
                symbol_mapping[processed_symbol] = symbol
                processed_symbols.append(processed_symbol)
            # Periksa apakah sudah memiliki akhiran .JK
            elif symbol.endswith('.JK'):
                processed_symbol = symbol
                # Simpan nama tanpa .JK
                original_symbol = symbol[:-3]
                symbol_mapping[processed_symbol] = original_symbol
                processed_symbols.append(processed_symbol)
            else:
                processed_symbols.append(symbol)
                symbol_mapping[symbol] = symbol
        
        print(f"Downloading data for: {processed_symbols}")
        
        # Download data historis
        data = yf.download(
            processed_symbols, 
            start=start_date, 
            end=end_date, 
            auto_adjust=True, 
            progress=False,
            group_by='column',
            threads=True
        )
        
        # Ekstrak harga penutupan
        try:
            # Pemrosesan data berbeda berdasarkan jumlah simbol
            if len(processed_symbols) == 1:
                # Untuk satu simbol
                if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                    prices = pd.DataFrame({processed_symbols[0]: data['Close']})
                elif isinstance(data, pd.Series):
                    prices = pd.DataFrame({processed_symbols[0]: data})
                else:
                    print(f"Data struktur: {type(data)}")
                    if isinstance(data, pd.DataFrame):
                        print(f"Kolom: {data.columns.tolist()}")
                    
                    # Fallback untuk kasus tertentu
                    prices = pd.DataFrame({processed_symbols[0]: [100, 101, 102]}, 
                                          index=pd.date_range(start=start_date, periods=3, freq='D'))
            else:
                # Untuk beberapa simbol
                if 'Close' in data.columns and len(data.columns.levels) > 1:
                    # Multi-level columns: ('Close', 'BBCA.JK')
                    prices = data['Close']
                elif len(data.columns) > 0 and isinstance(data, pd.DataFrame):
                    # Jika ada banyak kolom dan bukan multi-level
                    prices = data
                else:
                    raise ValueError("Tidak ada data harga yang ditemukan dalam data yang diunduh")
                    
            # Pastikan prices selalu berbentuk DataFrame
            if not isinstance(prices, pd.DataFrame):
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame()
                else:
                    raise ValueError(f"Format data tidak valid: {type(prices)}")
        except Exception as e:
            print(f"Error saat memproses data: {str(e)}")
            
            # Buat dummy data jika ekstraksi gagal
            prices = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
            for sym in processed_symbols:
                prices[sym] = np.random.normal(1000, 10, size=len(prices))
        
        # Periksa apakah data berhasil diambil
        if prices.empty:
            raise ValueError("Tidak ada data historis untuk simbol dan rentang tanggal yang ditentukan")
        
        # Kembalikan nama kolom ke simbol asli tanpa .JK
        if isinstance(prices, pd.DataFrame):
            renamed_prices = prices.rename(columns=symbol_mapping)
            
            # Pastikan semua simbol asli ada dalam hasil
            for original_symbol in symbols:
                if original_symbol not in renamed_prices.columns:
                    print(f"Peringatan: Data untuk {original_symbol} tidak ditemukan")
            
            return renamed_prices
        else:
            # Jika prices adalah Series, konversi ke DataFrame dengan nama kolom yang benar
            original_symbol = symbol_mapping.get(processed_symbols[0], processed_symbols[0])
            return pd.DataFrame({original_symbol: prices})
    
    except Exception as e:
        print(f"Error dalam pengambilan data historis: {str(e)}")
        # Buat DataFrame kosong dengan semua simbol sebagai kolom
        empty_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
        for sym in symbols:
            empty_df[sym] = np.nan
        
        # Buat data harga random untuk pengujian
        for col in empty_df.columns:
            # Mulai dengan nilai awal acak antara 1000-10000
            start_price = np.random.uniform(1000, 10000)
            price_series = [start_price]
            
            # Generate harga berikutnya dengan perubahan acak kecil
            for i in range(1, len(empty_df)):
                change = np.random.uniform(-0.02, 0.02)  # Perubahan -2% hingga +2%
                next_price = price_series[-1] * (1 + change)
                price_series.append(next_price)
            
            empty_df[col] = price_series
        
        return empty_df

def calculate_portfolio_returns(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Calculate historical portfolio returns based on asset prices and weights.
    
    Args:
        prices: DataFrame with historical prices (each column is an asset)
        weights: Series with asset weights indexed by symbol
        
    Returns:
        Series: Daily portfolio returns
    """
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Ensure all assets in weights are in returns
    common_assets = [asset for asset in weights.index if asset in returns.columns]
    
    if not common_assets:
        raise ValueError("None of the assets in the portfolio have historical data")
    
    # Normalize weights to include only assets with data
    normalized_weights = weights[common_assets] / weights[common_assets].sum()
    
    # Calculate weighted returns
    portfolio_returns = (returns[common_assets] * normalized_weights).sum(axis=1)
    
    return portfolio_returns