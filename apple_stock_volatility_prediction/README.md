# Apple Stock Volatility Prediction

This project predicts the hourly volatility of Apple (AAPL) stock using an LSTM+CNN hybrid model and includes backtesting.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create directories: `data/raw/`, `data/processed/`, `models/`
3. Run the main script: `python main.py`

## Files
- `main.py`: Main entry point.
- `scripts/data_crawler.py`: Crawls AAPL hourly data.
- `scripts/data_preprocessor.py`: Preprocesses data and generates features.
- `scripts/model_trainer.py`: Trains the LSTM+CNN model.
- `scripts/backtester.py`: Runs backtest and evaluates strategy.

## Notes
- Ensure internet connection for data crawling.
- Adjust `seq_length` and model parameters for better performance.