import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import gc
import sys

# Set memory growth for GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
except Exception as e:
    print(f"GPU memory config error: {e}")

class TimeSeriesGenerator(Sequence):
    """
    Custom data generator for time series data
    """
    def __init__(self, data, feature_cols, target_col, seq_length, batch_size=32, shuffle=True):
        """
        Initialize the generator
        
        Args:
            data (pd.DataFrame): DataFrame with features and target
            feature_cols (list): List of feature column names
            target_col (str): Name of the target column
            seq_length (int): Length of the sequence
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create valid indices (those that have enough preceding data points)
        self.valid_indices = list(range(seq_length, len(data)))
        self.indices = self.valid_indices.copy()
        
        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Return the number of batches"""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indices), self.seq_length, len(self.feature_cols)))
        batch_y = np.zeros((len(batch_indices),))
        
        # Fill batch arrays
        for i, idx in enumerate(batch_indices):
            # Get sequence
            seq_start = idx - self.seq_length
            seq_end = idx
            
            # Extract features and target
            batch_x[i] = self.data[self.feature_cols].values[seq_start:seq_end]
            batch_y[i] = self.data[self.target_col].values[idx]
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_processed_data(filepath, sample_size=None):
    """
    Load processed data from CSV file
    
    Args:
        filepath (str): Path to the processed data file
        sample_size (float): Fraction of data to use (0.0-1.0), None for all data
        
    Returns:
        pd.DataFrame: Processed data
    """
    print(f"Loading processed data from {filepath}...")
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Data loaded successfully with shape: {data.shape}")
        
        # Take a sample if specified
        if sample_size is not None and 0.0 < sample_size < 1.0:
            # Ensure we have enough data for a meaningful sample
            min_sample_size = 1000
            sample_rows = max(min_sample_size, int(len(data) * sample_size))
            
            # Take a stratified sample based on time to maintain temporal patterns
            # We'll divide the data into segments and sample from each segment
            num_segments = 10
            segment_size = len(data) // num_segments
            
            sampled_data = pd.DataFrame()
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(data)
                segment = data.iloc[start_idx:end_idx]
                
                # Sample from this segment
                segment_sample_size = int(sample_rows / num_segments)
                if len(segment) > segment_sample_size:
                    segment_sample = segment.sample(segment_sample_size)
                    sampled_data = pd.concat([sampled_data, segment_sample])
            
            # Sort by index to maintain time order
            sampled_data = sampled_data.sort_index()
            
            print(f"Using {len(sampled_data)} samples ({sample_size:.1%} of original data)")
            data = sampled_data
        
        # Check for NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in the data")
            # Drop rows with NaN values
            data = data.dropna()
            print(f"After dropping NaN values, data shape: {data.shape}")
            
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def build_lstm_cnn_model(input_shape, output_units=1):
    """
    Build a simpler hybrid LSTM+CNN model with fewer parameters
    
    Args:
        input_shape (tuple): Shape of the input data
        output_units (int): Number of output units
        
    Returns:
        Model: Keras model
    """
    print("Building LSTM+CNN hybrid model...")
    print(f"Input shape: {input_shape}")
    
    try:
        # Input layer
        inputs = Input(shape=input_shape, name="input_layer")
        
        # CNN branch - simpler with fewer filters
        cnn = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        
        # LSTM branch - simpler with fewer units
        lstm = LSTM(16)(inputs)
        
        # Merge CNN and LSTM branches
        merged = Concatenate()([cnn, lstm])
        
        # Dense layers - simpler with fewer units
        dense = Dense(16, activation='relu')(merged)
        
        # Output layer
        outputs = Dense(output_units)(dense)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Print model summary
        model.summary()
        
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        sys.exit(1)

def train_model_with_generator(train_gen, val_gen, model_path, epochs=30):
    """
    Train the model using data generators
    
    Args:
        train_gen (Sequence): Training data generator
        val_gen (Sequence): Validation data generator
        model_path (str): Path to save the model
        epochs (int): Number of epochs
        
    Returns:
        Model: Trained Keras model
        dict: Training history
    """
    print("Training the model using data generators...")
    
    try:
        # Get input shape from the generator
        sample_batch = train_gen[0]
        input_shape = sample_batch[0].shape[1:]
        
        # Build the model
        model = build_lstm_cnn_model(input_shape)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            model_path, 
            monitor='val_loss', 
            save_best_only=True,
            verbose=1
        )
        
        # Train the model - removed use_multiprocessing parameter
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        return model, history.history
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)

def plot_training_history(history):
    """Plot the training history"""
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Error plotting training history: {e}")

def evaluate_model_with_generator(model, test_gen):
    """
    Evaluate the model using a data generator
    
    Args:
        model (Model): Trained Keras model
        test_gen (Sequence): Test data generator
        
    Returns:
        float: Test loss
        float: Test MAE
    """
    print("Evaluating the model...")
    
    try:
        # Evaluate the model
        results = model.evaluate(test_gen, verbose=1)
        loss, mae = results
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # Make predictions on a sample batch
        sample_batch = test_gen[0]
        X_sample, y_sample = sample_batch
        
        y_pred = model.predict(X_sample)
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(y_sample, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs Predicted Volatility (Sample Batch)')
        plt.xlabel('Sample')
        plt.ylabel('Volatility')
        plt.legend()
        plt.savefig('prediction_results.png')
        print("Prediction results plot saved as 'prediction_results.png'")
        
        return loss, mae
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None, None

def train_volatility_model(data_filepath, model_filepath, sample_size=0.3, use_generator=True):
    """
    Main function to train the volatility model
    
    Args:
        data_filepath (str): Path to the processed data file
        model_filepath (str): Path to save the model
        sample_size (float): Fraction of data to use (0.0-1.0), None for all data
        use_generator (bool): Whether to use a data generator
    """
    # Load the processed data
    data = load_processed_data(data_filepath, sample_size=sample_size)
    
    # Define feature columns and target column
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
    print(f"Using features: {feature_cols}")
    
    target_col = 'hist_volatility'
    if target_col not in data.columns:
        print(f"Error: Target column '{target_col}' not found in data")
        sys.exit(1)
    
    # Split the data
    train_data, temp_data = train_test_split(data, test_size=0.2, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Sequence length and batch size
    seq_length = 24
    batch_size = 32
    
    if use_generator:
        # Create data generators
        train_gen = TimeSeriesGenerator(
            train_data, feature_cols, target_col, 
            seq_length=seq_length, batch_size=batch_size
        )
        
        val_gen = TimeSeriesGenerator(
            val_data, feature_cols, target_col, 
            seq_length=seq_length, batch_size=batch_size
        )
        
        test_gen = TimeSeriesGenerator(
            test_data, feature_cols, target_col, 
            seq_length=seq_length, batch_size=batch_size, shuffle=False
        )
        
        # Train the model
        model, history = train_model_with_generator(
            train_gen, val_gen, model_filepath, epochs=30
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate the model
        evaluate_model_with_generator(model, test_gen)
    else:
        # Use the original approach with full data loading
        # This is kept for comparison but not recommended for large datasets
        print("WARNING: Using full data loading approach, which may cause memory issues")
        
        # Create sequences
        X_train, y_train = [], []
        for i in range(len(train_data) - seq_length):
            X_train.append(train_data[feature_cols].values[i:i+seq_length])
            y_train.append(train_data[target_col].values[i+seq_length])
        
        X_val, y_val = [], []
        for i in range(len(val_data) - seq_length):
            X_val.append(val_data[feature_cols].values[i:i+seq_length])
            y_val.append(val_data[target_col].values[i+seq_length])
        
        X_test, y_test = [], []
        for i in range(len(test_data) - seq_length):
            X_test.append(test_data[feature_cols].values[i:i+seq_length])
            y_test.append(test_data[target_col].values[i+seq_length])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        # Build and train the model
        model = build_lstm_cnn_model(X_train.shape[1:])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            model_filepath, 
            monitor='val_loss', 
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history.history)
        
        # Evaluate the model
        loss, mae = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
    
    return model

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train volatility model')
    parser.add_argument('--sample-size', type=float, default=0.3,
                        help='Fraction of data to use (0.0-1.0), 1.0 for all data')
    parser.add_argument('--no-generator', action='store_true',
                        help='Do not use data generator (may cause memory issues)')
    args = parser.parse_args()
    
    # Train the volatility model
    train_volatility_model(
        "data/processed/aapl_features.csv", 
        "models/lstm_cnn_model.h5",
        sample_size=args.sample_size,
        use_generator=not args.no_generator
    )