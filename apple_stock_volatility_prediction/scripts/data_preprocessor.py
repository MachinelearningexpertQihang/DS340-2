import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load data from CSV file and convert columns to appropriate types"""
    print(f"Loading data from {filepath}...")
    
    # First, read the CSV without parsing dates to check column types
    df = pd.read_csv(filepath)
    
    # Convert index to datetime
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.set_index('Datetime', inplace=True)
    else:
        # If the index is already set, convert it to datetime
        df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Convert price and volume columns to numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Data loaded with shape: {df.shape}")
    print(f"Column dtypes: {df.dtypes}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")
    
    # Drop rows with NaN in essential columns
    essential_cols = ['Open', 'High', 'Low', 'Close']
    essential_cols = [col for col in essential_cols if col in df.columns]
    if essential_cols:
        df = df.dropna(subset=essential_cols)
        print(f"Shape after dropping NaN in essential columns: {df.shape}")
    
    return df

def calculate_volatility(data, window=24):
    """
    Calculate volatility metrics
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        window (int): Window size for volatility calculation (in hours)
        
    Returns:
        pd.DataFrame: DataFrame with added volatility metrics
    """
    print("Calculating volatility metrics...")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure all price data is numeric
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Historical volatility (standard deviation of log returns)
    df['hist_volatility'] = df['log_return'].rolling(window=window).std() * np.sqrt(252*24)
    
    # ATR (Average True Range) - a measure of volatility
    # First, calculate True Range
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['low_close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # Then calculate ATR
    df['atr'] = df['true_range'].rolling(window=window).mean()
    
    # Normalized ATR (ATR/Close price)
    df['normalized_atr'] = df['atr'] / df['Close']
    
    # Garman-Klass volatility estimator
    df['garman_klass'] = 0.5 * np.log(df['High'] / df['Low'])**2 - (2*np.log(2)-1) * np.log(df['Close'] / df['Open'])**2
    df['garman_klass_vol'] = np.sqrt(df['garman_klass'].rolling(window=window).mean() * 252*24)
    
    return df

def calculate_rsi(data, window=14):
    """Calculate RSI (Relative Strength Index)"""
    # Ensure data is numeric
    price_data = pd.to_numeric(data, errors='coerce')
    
    delta = price_data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rs = rs.fillna(0)
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Ensure data is numeric
    price_data = pd.to_numeric(data, errors='coerce')
    
    ema_fast = price_data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = price_data.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    # Ensure data is numeric
    price_data = pd.to_numeric(data, errors='coerce')
    
    sma = price_data.rolling(window=window).mean()
    std = price_data.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band

def add_technical_indicators(data):
    """
    Add technical indicators as features
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    print("Adding technical indicators...")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure Close price is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # RSI (Relative Strength Index)
    df['rsi'] = calculate_rsi(df['Close'])
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle
    
    # Moving Averages
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    
    # Price relative to moving averages
    df['price_to_sma_20'] = df['Close'] / df['sma_20']
    df['price_to_sma_50'] = df['Close'] / df['sma_50']
    df['price_to_sma_200'] = df['Close'] / df['sma_200']
    
    # Volume indicators
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    
    return df

def normalize_features(data, target_col='hist_volatility'):
    """
    Normalize features and prepare data for model training
    
    Args:
        data (pd.DataFrame): DataFrame with features
        target_col (str): Name of the target column
        
    Returns:
        pd.DataFrame: DataFrame with normalized features
        np.array: Scaler object for the features
    """
    print("Normalizing features...")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features and target
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'atr', 'normalized_atr', 'garman_klass_vol',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_width', 'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
        'volume_ratio'
    ]
    
    # Add hour and day_of_week if they exist
    if 'hour' in df.columns:
        feature_cols.append('hour')
    if 'day_of_week' in df.columns:
        feature_cols.append('day_of_week')
    
    # Filter to only include columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Create a scaler for the features
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # The target column is not normalized as it's what we want to predict
    
    return df, scaler

def preprocess_data(input_filepath, output_filepath):
    """
    Main function to preprocess the data
    
    Args:
        input_filepath (str): Path to the raw data file
        output_filepath (str): Path to save the processed data
    """
    # Load the data
    data = load_data(input_filepath)
    
    # Calculate volatility metrics
    data = calculate_volatility(data)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Normalize features
    normalized_data, scaler = normalize_features(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Save the processed data
    normalized_data.to_csv(output_filepath)
    print(f"Processed data saved to {output_filepath}")
    
    # Display sample of the processed data
    print("\nSample of the processed data:")
    print(normalized_data.head())
    
    return normalized_data, scaler

if __name__ == "__main__":
    # Preprocess the data
    preprocess_data("data/raw/aapl_hourly_10y.csv", "data/processed/aapl_features.csv")