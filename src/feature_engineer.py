import pandas as pd
import numpy as np
from .config import TECH_INDICATORS


class FeatureEngineer:
    """
    Adds technical indicators and engineering features to stock data
    """

    def __init__(self, dataframe):
        """
        Initialize with a DataFrame containing OHLCV data

        Args:
            dataframe: pandas DataFrame with columns [Open, High, Low, Close, Volume]
        """
        self.df = dataframe.copy()

    def add_basic_indicators(self):
        """
        Add basic momentum and volume indicators using manual calculations
        (Fallback if pandas-ta is not available)
        """
        # Daily returns
        self.df['daily_return'] = self.df['Close'].pct_change()

        # Volatility (20-day rolling std)
        self.df['volatility_20'] = self.df['daily_return'].rolling(20).std()

        # Volume indicators
        vol_ma_5 = self.df['Volume'].rolling(5).mean()
        vol_ma_20 = self.df['Volume'].rolling(20).mean()
        self.df['volume_ma_5'] = vol_ma_5
        self.df['volume_ma_20'] = vol_ma_20
        self.df['volume_ratio'] = self.df['Volume'] / self.df['volume_ma_20']

        # Price-based indicators
        self.df['price_range'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['close_to_open'] = (self.df['Close'] - self.df['Open']) / self.df['Open']

        return self.df

    def add_technical_indicators(self):
        """
        Add advanced technical indicators using pandas-ta
        """
        try:
            import pandas_ta as ta

            # Moving Averages
            self.df.ta.sma(length=10, append=True)
            self.df.ta.sma(length=20, append=True)
            self.df.ta.sma(length=50, append=True)

            # Momentum Indicators
            self.df.ta.rsi(length=14, append=True)
            self.df.ta.macd(append=True)  # MACD, MACDh, MACDs
            self.df.ta.stoch(append=True)  # STOCHk, STOCHd

            # Volatility Indicators
            self.df.ta.bbands(append=True)  # BBL, BBM, BBU, BBB, BBP

            # Volume Indicators
            self.df.ta.obv(append=True)  # On Balance Volume

            print("✅ Added advanced technical indicators with pandas-ta")

        except ImportError:
            print("⚠️ pandas-ta not available, using basic indicators only")
            self.add_basic_indicators()

        return self.df

    def add_custom_features(self):
        """
        Add custom engineered features specifically for stock prediction
        """
        # Trend features
        self.df['price_momentum'] = self.df['Close'] / self.df['Close'].shift(5) - 1

        # Mean reversion features
        self.df['close_vs_sma20'] = self.df['Close'] / self.df.get('SMA_20', self.df['Close']) - 1
        self.df['close_vs_sma50'] = self.df['Close'] / self.df.get('SMA_50', self.df['Close']) - 1

        # Lag features (previous days)
        for lag in [1, 3, 5, 10]:
            self.df[f'close_lag_{lag}'] = self.df['Close'].shift(lag)
            self.df[f'return_lag_{lag}'] = self.df['daily_return'].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            self.df[f'close_rolling_mean_{window}'] = self.df['Close'].rolling(window).mean()
            self.df[f'close_rolling_std_{window}'] = self.df['Close'].rolling(window).std()

        return self.df

    def process_data(self):
        """
        Complete feature engineering pipeline

        Returns:
            tuple: (features_dataframe, target_dataframe)
        """
        # Add basic indicators first (these are always available)
        self.add_basic_indicators()

        # Add advanced indicators
        self.add_technical_indicators()

        # Add custom engineered features
        self.add_custom_features()

        # Remove rows with NaN values created by indicators/lags
        self.df = self.df.dropna()

        print(f"Processed {len(self.df)} rows with {len(self.df.columns)} features")

        return self.df

    def prepare_features_and_target(self, prediction_days=1):
        """
        Prepare features and target for ML training

        Args:
            prediction_days: How many days ahead to predict (default: next day)

        Returns:
            tuple: (features X, target y)
        """
        # Process data through feature engineering
        df_processed = self.process_data()

        # Create target (future price)
        df_processed['target'] = df_processed['Close'].shift(-prediction_days)

        # Remove rows with NaN target
        df_processed = df_processed.dropna()

        # Select features (exclude OHLCV and target)
        exclude_columns = ['target', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]

        X = df_processed[feature_columns]
        y = df_processed['target']

        print(f"Prepared {len(feature_columns)} features and {len(y)} target values")

        return X, y

    def get_feature_importance_template(self):
        """
        Returns a template of feature groups for importance analysis
        """
        return {
            'price_features': ['daily_return', 'close_vs_sma20', 'close_vs_sma50'],
            'volume_features': ['volume_ma_5', 'volume_ma_20', 'volume_ratio'],
            'technical_indicators': ['RSI_14', 'MACD', 'STOCHk'],
            'lag_features': [col for col in self.df.columns if 'lag_' in col],
            'momentum_features': ['price_momentum', 'close_to_open']
        }
