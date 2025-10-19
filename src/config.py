# Configuration settings for AI Stock Predictor

# Default stocks to track
DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# Training settings
TRAINING_DAYS = 500  # About 2 years of data
TEST_SPLIT = 0.2     # 20% of data for testing
RANDOM_STATE = 42    # For reproducible results

# Prediction settings
DEFAULT_PREDICTION_DAYS = 1  # Predict next day by default
MAX_PREDICTION_DAYS = 30     # Maximum days to predict ahead

# Data settings
DATA_CACHE_DAYS = 30   # Cache data for 30 days to avoid redundant API calls
TECH_INDICATORS = ['SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'BBANDS', 'STOCH']

# Model settings
MODEL_ACCURACY_THRESHOLD = 0.7  # R² threshold for acceptable model performance
