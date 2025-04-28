import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import os
import sys
import gc
import time

def load_data_and_model(data_filepath, model_filepath, sample_size=None):
    """
    Load the processed data and trained model
    
    Args:
        data_filepath (str): Path to the processed data file
        model_filepath (str): Path to the trained model
        sample_size (float): Fraction of data to use (0.0-1.0), None for all data
        
    Returns:
        pd.DataFrame: Processed data
        Model: Trained model
    """
    print(f"Loading data from {data_filepath}...")
    try:
        data = pd.read_csv(data_filepath, index_col=0, parse_dates=True)
        print(f"Data loaded successfully with shape: {data.shape}")
        
        # Take a sample if specified
        if sample_size is not None and 0.0 < sample_size < 1.0:
            # Use the most recent data for backtesting
            sample_rows = int(len(data) * sample_size)
            data = data.iloc[-sample_rows:]
            print(f"Using {len(data)} samples ({sample_size:.1%} of original data)")
        
        # Check for NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in the data")
            # Drop rows with NaN values
            data = data.dropna()
            print(f"After dropping NaN values, data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Loading model from {model_filepath}...")
    try:
        # Define custom objects to handle serialization issues
        custom_objects = {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError()
        }
        
        # Try to load the model with custom objects
        model = load_model(model_filepath, custom_objects=custom_objects)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to rebuild model and load weights...")
        
        try:
            # If loading fails, try to rebuild the model and load weights only
            from scripts.model_trainer import build_lstm_cnn_model
            
            # Get input shape from the data
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume', 
                'atr', 'normalized_atr', 'garman_klass_vol',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
                'volume_ratio'
            ]
            
            # Add hour and day_of_week if they exist
            if 'hour' in data.columns:
                feature_cols.append('hour')
            if 'day_of_week' in data.columns:
                feature_cols.append('day_of_week')
            
            # Filter to only include columns that exist in the dataframe
            feature_cols = [col for col in feature_cols if col in data.columns]
            
            # Rebuild model with the same architecture
            seq_length = 24
            input_shape = (seq_length, len(feature_cols))
            model = build_lstm_cnn_model(input_shape)
            
            # Load weights only
            model.load_weights(model_filepath)
            print("Model rebuilt and weights loaded successfully")
        except Exception as e2:
            print(f"Error rebuilding model: {e2}")
            sys.exit(1)
    
    return data, model

def implement_simple_ma_strategy(data, short_window=5, long_window=20):
    """
    Implement a simple moving average crossover strategy
    
    Args:
        data (pd.DataFrame): Data with price information
        short_window (int): Short moving average window
        long_window (int): Long moving average window
        
    Returns:
        pd.DataFrame: Data with trading signals
    """
    print("Implementing simple moving average crossover strategy...")
    
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate moving averages
        df['short_ma'] = df['Close'].rolling(window=short_window).mean()
        df['long_ma'] = df['Close'].rolling(window=long_window).mean()
        
        # Generate trading signals
        df['signal'] = 0  # 0: no position, 1: long, -1: short
        
        # Buy signal: short MA crosses above long MA
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
        
        # Sell signal: short MA crosses below long MA
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
        
        # Print signal distribution
        signal_counts = df['signal'].value_counts()
        print(f"Signal distribution: {signal_counts.to_dict()}")
        
        return df
    except Exception as e:
        print(f"Error implementing strategy: {e}")
        sys.exit(1)

def calculate_returns(data):
    """
    Calculate returns based on trading signals
    
    Args:
        data (pd.DataFrame): Data with trading signals
        
    Returns:
        pd.DataFrame: Data with returns
    """
    print("Calculating returns...")
    
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Calculate strategy returns
        df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
        
        # Print sample of returns
        print("\nSample of returns:")
        print(df[['Close', 'signal', 'daily_return', 'strategy_return']].head(10))
        
        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        df['strategy_cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
        
        # Drop NaN values
        df = df.dropna()
        
        # Print summary statistics
        print("\nReturn statistics:")
        print(df[['daily_return', 'strategy_return']].describe())
        
        return df
    except Exception as e:
        print(f"Error calculating returns: {e}")
        sys.exit(1)

def calculate_performance_metrics(data):
    """
    Calculate performance metrics
    
    Args:
        data (pd.DataFrame): Data with returns
        
    Returns:
        dict: Performance metrics
    """
    print("Calculating performance metrics...")
    
    try:
        # Check if we have enough data
        if len(data) < 2:
            print("Error: Not enough data to calculate performance metrics")
            return {}
        
        # Calculate total return
        buy_hold_return = data['cumulative_return'].iloc[-1]
        strategy_return = data['strategy_cumulative_return'].iloc[-1]
        
        print(f"Buy & Hold final cumulative return: {buy_hold_return:.4f}")
        print(f"Strategy final cumulative return: {strategy_return:.4f}")
        
        # Calculate annualized return
        # Use actual number of trading days instead of calendar days
        trading_days = len(data)
        ann_factor = 252 / trading_days
        
        ann_buy_hold_return = (1 + buy_hold_return) ** ann_factor - 1
        ann_strategy_return = (1 + strategy_return) ** ann_factor - 1
        
        # Calculate volatility
        buy_hold_vol = data['daily_return'].std() * np.sqrt(252)
        strategy_vol = data['strategy_return'].std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        buy_hold_sharpe = (ann_buy_hold_return - risk_free_rate) / buy_hold_vol if buy_hold_vol > 0 else 0
        strategy_sharpe = (ann_strategy_return - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        
        # Calculate maximum drawdown
        buy_hold_cum_returns = data['cumulative_return']
        strategy_cum_returns = data['strategy_cumulative_return']
        
        # Calculate drawdown safely
        buy_hold_peak = buy_hold_cum_returns.cummax()
        strategy_peak = strategy_cum_returns.cummax()
        
        # Avoid division by zero
        buy_hold_drawdown = np.where(buy_hold_peak == 0, 0, (buy_hold_cum_returns - buy_hold_peak) / buy_hold_peak)
        strategy_drawdown = np.where(strategy_peak == 0, 0, (strategy_cum_returns - strategy_peak) / strategy_peak)
        
        buy_hold_max_drawdown = np.min(buy_hold_drawdown) if len(buy_hold_drawdown) > 0 else 0
        strategy_max_drawdown = np.min(strategy_drawdown) if len(strategy_drawdown) > 0 else 0
        
        # Calculate trade statistics
        # Count position changes as trades
        position_changes = data['signal'].diff().fillna(0)
        trades = position_changes != 0
        num_trades = trades.sum()
        
        # Calculate trade returns
        trade_returns = []
        current_position = 0
        entry_price = 0
        
        for i in range(len(data)):
            if position_changes.iloc[i] != 0:
                # Position changed, a trade occurred
                if current_position != 0:
                    # Close previous position
                    exit_price = data['Close'].iloc[i]
                    trade_return = (exit_price / entry_price - 1) * current_position
                    trade_returns.append(trade_return)
                    print(f"Trade closed: Position {current_position}, Entry {entry_price:.2f}, Exit {exit_price:.2f}, Return {trade_return:.4f}")
                
                # Open new position
                current_position = data['signal'].iloc[i]
                if current_position != 0:
                    entry_price = data['Close'].iloc[i]
        
        # Calculate win rate
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
            profit_factor = -sum(r for r in trade_returns if r > 0) / sum(r for r in trade_returns if r <= 0) if sum(r for r in trade_returns if r <= 0) < 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Create a dictionary with the metrics
        metrics = {
            'Buy & Hold Return': buy_hold_return,
            'Strategy Return': strategy_return,
            'Annualized Buy & Hold Return': ann_buy_hold_return,
            'Annualized Strategy Return': ann_strategy_return,
            'Buy & Hold Volatility': buy_hold_vol,
            'Strategy Volatility': strategy_vol,
            'Buy & Hold Sharpe Ratio': buy_hold_sharpe,
            'Strategy Sharpe Ratio': strategy_sharpe,
            'Buy & Hold Max Drawdown': buy_hold_max_drawdown,
            'Strategy Max Drawdown': strategy_max_drawdown,
            'Win Rate': win_rate,
            'Number of Trades': num_trades,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {}

def plot_backtest_results(data, metrics):
    """
    Plot backtest results
    
    Args:
        data (pd.DataFrame): Data with returns
        metrics (dict): Performance metrics
    """
    print("Plotting backtest results...")
    
    try:
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(data['cumulative_return'], label='Buy & Hold')
        plt.plot(data['strategy_cumulative_return'], label='Strategy')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        
        # Plot trading signals
        plt.subplot(2, 1, 2)
        plt.plot(data['Close'], label='Close Price')
        plt.plot(data['short_ma'], label='Short MA', linestyle='--')
        plt.plot(data['long_ma'], label='Long MA', linestyle='--')
        
        # Plot buy and sell signals
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal')
        
        plt.title('Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        print("Backtest results plot saved as 'backtest_results.png'")
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    except Exception as e:
        print(f"Error plotting results: {e}")

def run_simple_backtest(data_filepath, sample_size=0.1, short_window=5, long_window=20):
    """
    Run a simple moving average crossover backtest
    
    Args:
        data_filepath (str): Path to the processed data file
        sample_size (float): Fraction of data to use (0.0-1.0), None for all data
        short_window (int): Short moving average window
        long_window (int): Long moving average window
    """
    print("Running simple moving average crossover backtest...")
    
    # Load data
    try:
        data = pd.read_csv(data_filepath, index_col=0, parse_dates=True)
        print(f"Data loaded successfully with shape: {data.shape}")
        
        # Take a sample if specified
        if sample_size is not None and 0.0 < sample_size < 1.0:
            # Use the most recent data for backtesting
            sample_rows = int(len(data) * sample_size)
            data = data.iloc[-sample_rows:]
            print(f"Using {len(data)} samples ({sample_size:.1%} of original data)")
        
        # Check for NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in the data")
            # Drop rows with NaN values
            data = data.dropna()
            print(f"After dropping NaN values, data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Implement simple moving average crossover strategy
    data_with_signals = implement_simple_ma_strategy(data, short_window, long_window)
    
    # Calculate returns
    data_with_returns = calculate_returns(data_with_signals)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(data_with_returns)
    
    # Plot backtest results
    plot_backtest_results(data_with_returns, metrics)
    
    return data_with_returns

def run_backtest(data_filepath, model_filepath, sample_size=0.1, window=20, k=1.5):
    """
    Main function to run the backtest
    
    Args:
        data_filepath (str): Path to the processed data file
        model_filepath (str): Path to the trained model
        sample_size (float): Fraction of data to use (0.0-1.0), None for all data
        window (int): Window size for the moving average
        k (float): Multiplier for the volatility
    """
    print("Running simple backtest instead of model-based backtest...")
    return run_simple_backtest(data_filepath, sample_size, short_window=5, long_window=window)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--data-file', type=str, default="data/processed/aapl_features.csv",
                        help='Path to the processed data file')
    parser.add_argument('--model-file', type=str, default="models/lstm_cnn_model.h5",
                        help='Path to the trained model')
    parser.add_argument('--sample-size', type=float, default=0.1,
                        help='Fraction of data to use (0.0-1.0), 1.0 for all data')
    parser.add_argument('--short-window', type=int, default=5,
                        help='Short moving average window')
    parser.add_argument('--long-window', type=int, default=20,
                        help='Long moving average window')
    args = parser.parse_args()
    
    # Run the backtest
    run_simple_backtest(
        args.data_file, 
        sample_size=args.sample_size,
        short_window=args.short_window,
        long_window=args.long_window
    )