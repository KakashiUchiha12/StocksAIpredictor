from .data_loader import StockDataLoader
from .feature_engineer import FeatureEngineer
from .model_trainer import StockPredictor
from .config import DEFAULT_STOCKS, DEFAULT_PREDICTION_DAYS, MAX_PREDICTION_DAYS
import pandas as pd
import os


class StockPredictionEngine:
    """
    Main engine that orchestrates the entire stock prediction workflow
    """

    def __init__(self):
        self.models = {}  # Cache loaded models
        self.stock_info = {}  # Cache stock metadata

    def train_single_model(self, ticker, days=500, force_refresh=False):
        """
        Train a model for a single stock

        Args:
            ticker: Stock ticker symbol
            days: Historical data period in days
            force_refresh: Whether to ignore data cache

        Returns:
            dict: Training results and metrics
        """
        print(f"\n{'='*50}")
        print(f"🎯 TRAINING MODEL FOR {ticker.upper()}")
        print(f"{'='*50}")

        try:
            # Load stock data
            loader = StockDataLoader(ticker, days)
            df = loader.get_data(force_refresh)

            if df.empty:
                raise ValueError(f"No data available for {ticker}")

            # Engineer features
            engineer = FeatureEngineer(df)
            features, target = engineer.prepare_features_and_target()

            # Train model
            predictor = StockPredictor(ticker)
            metrics = predictor.train(features, target)

            # Cache the model
            self.models[ticker] = predictor

            # Get stock information
            stock_info = self.get_stock_info(ticker)

            results = {
                'ticker': ticker,
                'success': True,
                'metrics': metrics,
                'training_samples': metrics['training_samples'],
                'test_samples': metrics['test_samples'],
                'stock_info': stock_info
            }

            print("✅ Training completed successfully!")
            return results

        except Exception as e:
            print(f"❌ Training failed for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'success': False,
                'error': str(e)
            }

    def train_all_models(self, tickers=DEFAULT_STOCKS, days=500, force_refresh=False):
        """
        Train models for multiple stocks

        Args:
            tickers: List of stock tickers
            days: Historical data period in days
            force_refresh: Whether to ignore data cache

        Returns:
            dict: Results for all stocks
        """
        results = {}
        successful = 0

        print(f"🤖 STARTING TRAINING FOR {len(tickers)} STOCKS")
        print("="*60)

        for ticker in tickers:
            training_result = self.train_single_model(ticker, days, force_refresh)
            results[ticker] = training_result

            if training_result.get('success', False):
                successful += 1
                metrics = training_result.get('metrics', {})
                r2 = metrics.get('r2', 0)
                mae = metrics.get('mae', 0)
                print(f"✅ {ticker}: R² = {r2:.3f}, MAE = ${mae:.2f}")
            else:
                print(f"❌ {ticker}: {training_result.get('error', 'Unknown error')}")

        print(f"\n{'='*60}")
        print(f"🏆 TRAINING COMPLETE: {successful}/{len(tickers)} models trained successfully")
        print(f"{'='*60}")

        return results

    def predict_stock(self, ticker, prediction_days=DEFAULT_PREDICTION_DAYS):
        """
        Make a prediction for a specific stock

        Args:
            ticker: Stock ticker symbol
            prediction_days: Days to predict ahead (1-30)

        Returns:
            dict: Prediction results with confidence intervals
        """
        if prediction_days < 1 or prediction_days > MAX_PREDICTION_DAYS:
            raise ValueError(f"Prediction days must be between 1 and {MAX_PREDICTION_DAYS}")

        try:
            # Load or get cached model
            predictor = self._load_model(ticker)

            # Get recent data for features (need enough for indicators)
            recent_days = max(100, prediction_days * 10)  # Ensure we have enough data
            loader = StockDataLoader(ticker, recent_days)
            recent_df = loader.get_data()

            if recent_df.empty:
                raise ValueError(f"No recent data available for {ticker}")

            # Engineer features for the most recent data
            engineer = FeatureEngineer(recent_df)
            features, _ = engineer.prepare_features_and_target(prediction_days)

            # Get the most recent feature row for prediction
            if len(features) == 0:
                raise ValueError(f"Not enough data for prediction after feature engineering")

            latest_features = features.iloc[-1:].copy()

            # Make prediction with confidence interval
            prediction_result = predictor.predict_with_confidence(latest_features)

            # Get current price for comparison
            current_price = recent_df['Close'].iloc[-1]

            # Calculate prediction vs current price
            predicted_price = prediction_result['prediction']
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100

            # Create comprehensive result
            result = {
                'ticker': ticker,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'prediction_days': prediction_days,
                'confidence_interval': prediction_result.get('confidence_interval'),
                'lower_bound': prediction_result.get('lower_bound'),
                'upper_bound': prediction_result.get('upper_bound'),
                'model_metrics': predictor.metrics,
                'stock_info': self.get_stock_info(ticker),
                'prediction_direction': 'UP ⬆️' if price_change > 0 else 'DOWN ⬇️' if price_change < 0 else 'FLAT ➡️',
                'timestamp': pd.Timestamp.now().isoformat()
            }

            return result

        except Exception as e:
            print(f"❌ Prediction failed for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'success': False
            }

    def predict_multiple_stocks(self, tickers=DEFAULT_STOCKS, prediction_days=DEFAULT_PREDICTION_DAYS):
        """
        Make predictions for multiple stocks

        Args:
            tickers: List of stock tickers
            prediction_days: Days to predict ahead

        Returns:
            dict: Predictions for all stocks
        """
        results = {}

        print(f"\n🔮 MAKING PREDICTIONS FOR {len(tickers)} STOCKS ({prediction_days} days ahead)")
        print("="*70)

        for ticker in tickers:
            prediction = self.predict_stock(ticker, prediction_days)
            results[ticker] = prediction

            if 'error' not in prediction:
                price = prediction.get('predicted_price', 0)
                change_pct = prediction.get('price_change_pct', 0)
                direction = prediction.get('prediction_direction', '')

                print(f"🎯 {ticker}: ${price:.2f} ({change_pct:+.1f}%) {direction}")
            else:
                print(f"❌ {ticker}: {prediction.get('error', 'Unknown error')}")

        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"\n✅ Generated {successful}/{len(tickers)} predictions successfully")

        return results

    def _load_model(self, ticker):
        """
        Load model from disk or cache

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockPredictor: Loaded model instance
        """
        ticker = ticker.upper()

        # Check if model is cached
        if ticker in self.models:
            return self.models[ticker]

        try:
            # Load from disk
            predictor = StockPredictor.load_model(ticker)
            self.models[ticker] = predictor
            return predictor

        except FileNotFoundError:
            # Model not trained yet
            raise ValueError(f"No trained model found for {ticker}. Please train the model first.")

    def get_stock_info(self, ticker):
        """
        Get cached stock information

        Args:
            ticker: Stock ticker symbol

        Returns:
            dict: Stock information
        """
        if ticker not in self.stock_info:
            from .data_loader import get_stock_info
            self.stock_info[ticker] = get_stock_info(ticker)

        return self.stock_info[ticker]

    def get_model_performance_summary(self, tickers=None):
        """
        Get performance summary for trained models

        Args:
            tickers: List of tickers to include (default: all cached models)

        Returns:
            dict: Performance summary
        """
        if tickers is None:
            tickers = list(self.models.keys())

        summary = {
            'total_models': len(tickers),
            'models': {}
        }

        for ticker in tickers:
            try:
                predictor = self._load_model(ticker)
                summary['models'][ticker] = predictor.metrics
            except:
                summary['models'][ticker] = {'error': 'Model not available'}

        return summary

    def clear_cache(self):
        """
        Clear all cached models and data
        """
        self.models.clear()
        self.stock_info.clear()
        print("🧹 Cache cleared")
