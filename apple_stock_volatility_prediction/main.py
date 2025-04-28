import os
import argparse
from scripts.data_crawler import fetch_apple_hourly_data, save_data
from scripts.data_preprocessor import preprocess_data
from scripts.model_trainer import train_volatility_model
from scripts.backtester import run_backtest

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Apple Stock Volatility Prediction")
    
    parser.add_argument("--fetch-data", action="store_true", help="Fetch Apple stock data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train-model", action="store_true", help="Train the volatility model")
    parser.add_argument("--backtest", action="store_true", help="Run the backtest")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    
    # Add new arguments for model training
    parser.add_argument("--sample-size", type=float, default=0.3,
                        help="Fraction of data to use for training (0.0-1.0), 1.0 for all data")
    parser.add_argument("--no-generator", action="store_true",
                        help="Do not use data generator (may cause memory issues)")
    
    # Add new arguments for backtesting
    parser.add_argument("--backtest-sample", type=float, default=0.1,
                        help="Fraction of data to use for backtesting (0.0-1.0), 1.0 for all data")
    parser.add_argument("--window", type=int, default=20,
                        help="Window size for the moving average in backtesting")
    parser.add_argument("--k", type=float, default=1.5,
                        help="Multiplier for the volatility in backtesting")
    
    return parser.parse_args()

def main():
    """Main function to run the entire pipeline"""
    # Create directories
    create_directories()
    
    # Parse arguments
    args = parse_arguments()
    
    # Define file paths
    raw_data_path = "data/raw/aapl_hourly_10y.csv"
    processed_data_path = "data/processed/aapl_features.csv"
    model_path = "models/lstm_cnn_model.h5"
    
    # Run the entire pipeline if --all is specified
    if args.all:
        args.fetch_data = True
        args.preprocess = True
        args.train_model = True
        args.backtest = True
    
    # Fetch data
    if args.fetch_data:
        print("\n=== Fetching Apple Stock Data ===")
        apple_data = fetch_apple_hourly_data(years=10)
        save_data(apple_data, raw_data_path)
    
    # Preprocess data
    if args.preprocess:
        print("\n=== Preprocessing Data ===")
        preprocess_data(raw_data_path, processed_data_path)
    
    # Train model
    if args.train_model:
        print("\n=== Training Volatility Model ===")
        train_volatility_model(
            processed_data_path, 
            model_path,
            sample_size=args.sample_size,
            use_generator=not args.no_generator
        )
    
    # Run backtest
    if args.backtest:
        print("\n=== Running Backtest ===")
        run_backtest(
            processed_data_path, 
            model_path,
            sample_size=args.backtest_sample,
            window=args.window,
            k=args.k
        )
    
    print("\n=== Pipeline Completed ===")

if __name__ == "__main__":
    main()