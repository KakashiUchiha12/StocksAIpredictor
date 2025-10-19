import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from .config import TRAINING_DAYS, DATA_CACHE_DAYS


class StockDataLoader:
    """
    Handles downloading and caching stock data from Yahoo Finance
    """

    def __init__(self, ticker, days=TRAINING_DAYS):
        self.ticker = ticker.upper()
        self.days = days
        self.cache_dir = "data"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_data(self, force_refresh=False):
        """
        Download stock data with caching to avoid redundant API calls

        Args:
            force_refresh: If True, ignore cache and download fresh data

        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        cache_file = f"{self.cache_dir}/{self.ticker}_data.pkl"

        # Check if cached data exists and is recent
        if not force_refresh and os.path.exists(cache_file):
            file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
            if file_age_days < DATA_CACHE_DAYS:
                print(f"Loading {self.ticker} data from cache...")
                return pd.read_pickle(cache_file)

        # Download fresh data
        print(f"Downloading {self.ticker} data from Yahoo Finance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)

        try:
            df = yf.download(
                self.ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

            # Handle MultiIndex columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns (remove ticker level)
                df.columns = df.columns.get_level_values(0)

            # Cache the data
            df.to_pickle(cache_file)
            print(f"✅ Downloaded {len(df)} days of {self.ticker} data")

            return df

        except Exception as e:
            print(f"❌ Error downloading {self.ticker}: {str(e)}")
            return pd.DataFrame()


def load_multiple_stocks(tickers, days=TRAINING_DAYS, force_refresh=False):
    """
    Load data for multiple stocks efficiently

    Args:
        tickers: List of stock tickers
        days: Number of days of historical data
        force_refresh: Whether to ignore cache

    Returns:
        dict: Dictionary mapping ticker to DataFrame
    """
    all_data = {}
    total_tickers = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{total_tickers}] Loading {ticker}...")
        loader = StockDataLoader(ticker, days)
        df = loader.get_data(force_refresh)
        if not df.empty:
            all_data[ticker] = df

    successful = len(all_data)
    if successful == total_tickers:
        print(f"\n✅ Successfully loaded data for all {total_tickers} stocks")
    else:
        print(f"\n⚠️ Loaded data for {successful}/{total_tickers} stocks")

    return all_data


def get_stock_info(ticker):
    """
    Get basic company information for a ticker

    Args:
        ticker: Stock ticker symbol

    Returns:
        dict: Company information or empty dict if failed
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap'),
            'currency': info.get('currency', 'USD')
        }
    except:
        return {'name': ticker, 'sector': 'Unknown'}
