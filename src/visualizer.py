import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class StockVisualizer:
    """
    Visualization tools for stock data and predictions
    """

    def __init__(self):
        self.style = 'seaborn-v0_8'  # Use consistent matplotlib style
        plt.style.use(self.style)

    def plot_stock_history(self, df, ticker, title=None, save_path=None):
        """
        Plot stock price history with volume

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker
            title: Optional plot title
            save_path: Path to save figure (optional)
        """
        if df.empty:
            print(f"No data to plot for {ticker}")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Price plot
        ax1.plot(df.index, df['Close'], linewidth=1.5, label='Close Price')
        ax1.fill_between(df.index, df['Low'], df['High'], alpha=0.1, color='blue', label='High-Low Range')
        ax1.set_title(title or f'{ticker} Stock Price History')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Volume plot
        ax2.bar(df.index, df['Volume'], alpha=0.7, color='orange', width=1)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to {save_path}")

        return fig

    def plot_prediction_comparison(self, df, predictions, ticker, days_ahead=1, save_path=None):
        """
        Plot actual vs predicted prices

        Args:
            df: Historical DataFrame
            predictions: Dict with prediction results
            ticker: Stock ticker
            days_ahead: Prediction horizon
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax.plot(df.index, df['Close'], 'b-', linewidth=1.5, label='Historical Price')

        # Add prediction point
        if ticker in predictions and 'error' not in predictions[ticker]:
            pred_data = predictions[ticker]
            last_date = df.index[-1]
            current_price = pred_data['current_price']
            predicted_price = pred_data['predicted_price']
            prediction_date = last_date + pd.Timedelta(days=days_ahead)

            # Plot current price marker
            ax.plot(last_date, current_price, 'ro', markersize=8, label='Current Price')

            # Plot prediction
            ax.plot(prediction_date, predicted_price, 'g^', markersize=10, label=f'{days_ahead}-Day Prediction')

            # Add error bars if available
            if pred_data.get('confidence_interval'):
                yerr = pred_data['confidence_interval'] / 2  # Half interval for upper/lower
                ax.errorbar(prediction_date, predicted_price, yerr=yerr,
                          fmt='none', ecolor='red', capsize=5, label='Confidence Interval')

            # Add annotation
            ax.annotate('.2f',
                       xy=(prediction_date, predicted_price),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        ax.set_title(f'{ticker} Stock Price: History vs {days_ahead}-Day Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to {save_path}")

        return fig

    def plot_model_performance(self, predictions, tickers=None, save_path=None):
        """
        Plot prediction accuracy across stocks

        Args:
            predictions: Dict of prediction results
            tickers: List of tickers to plot (optional)
            save_path: Path to save figure (optional)
        """
        if tickers is None:
            tickers = list(predictions.keys())

        # Filter successful predictions
        successful_preds = {
            ticker: pred for ticker, pred in predictions.items()
            if 'error' not in pred and ticker in tickers
        }

        if not successful_preds:
            print("No successful predictions to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stock Prediction Performance Analysis', fontsize=14)

        tickers_list = list(successful_preds.keys())
        prices = [successful_preds[t]['current_price'] for t in tickers_list]
        predicted = [successful_preds[t]['predicted_price'] for t in tickers_list]
        r2_scores = []
        directional_acc = []

        for ticker in tickers_list:
            metrics = successful_preds[ticker].get('model_metrics', {})
            r2_scores.append(metrics.get('r2', 0))
            directional_acc.append(metrics.get('directional_accuracy', 0))

        # Price comparison
        axes[0, 0].scatter(prices, predicted, alpha=0.7, s=50)
        for i, ticker in enumerate(tickers_list):
            axes[0, 0].annotate(ticker, (prices[i], predicted[i]), xytext=(5, 5), textcoords='offset points')

        # Perfect prediction line
        max_price = max(max(prices), max(predicted))
        min_price = min(min(prices), min(predicted))
        axes[0, 0].plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Current Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].set_title('Predicted vs Current Prices')
        axes[0, 0].grid(True, alpha=0.3)

        # Percentage change
        changes = [(predicted[i] - prices[i]) / prices[i] * 100 for i in range(len(prices))]
        axes[0, 1].bar(range(len(tickers_list)), changes, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xticks(range(len(tickers_list)))
        axes[0, 1].set_xticklabels(tickers_list, rotation=45)
        axes[0, 1].set_ylabel('Predicted Change (%)')
        axes[0, 1].set_title('Prediction Direction')
        axes[0, 1].grid(True, alpha=0.3)

        # R² Scores
        axes[1, 0].bar(range(len(tickers_list)), r2_scores, alpha=0.7, color='green')
        axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Good Performance (0.7)')
        axes[1, 0].set_xticks(range(len(tickers_list)))
        axes[1, 0].set_xticklabels(tickers_list, rotation=45)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Accuracy (R² Scores)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Directional Accuracy
        axes[1, 1].bar(range(len(tickers_list)), directional_acc, alpha=0.7, color='orange')
        axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[1, 1].set_xticks(range(len(tickers_list)))
        axes[1, 1].set_xticklabels(tickers_list, rotation=45)
        axes[1, 1].set_ylabel('Directional Accuracy (%)')
        axes[1, 1].set_title('Prediction Direction Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to {save_path}")

        return fig

    def plot_feature_importance(self, predictor, ticker, top_n=10, save_path=None):
        """
        Plot feature importance for a specific model

        Args:
            predictor: StockPredictor instance
            ticker: Stock ticker
            top_n: Number of top features to show
            save_path: Path to save figure (optional)
        """
        importance_dict = predictor.get_feature_importance()

        if not importance_dict:
            print(f"No feature importance data available for {ticker}")
            return

        # Get top N features
        sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        fig, ax = plt.subplots(figsize=(12, 8))

        features = [f[0] for f in sorted_features]
        weights = [abs(f[1]) for f in sorted_features]  # Absolute values for better visualization

        bars = ax.barh(range(len(features)), weights, alpha=0.7)

        # Color bars based on positive/negative influence
        for idx, (_, weight) in enumerate(sorted_features):
            color = 'green' if weight > 0 else 'red'
            bars[idx].set_color(color)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance (Absolute Coefficient)')
        ax.set_title(f'{ticker} - Top {top_n} Important Features for Price Prediction')
        ax.grid(True, alpha=0.3)

        # Add legend
        import matplotlib.patches as mpatches
        green_patch = mpatches.Patch(color='green', label='Positive influence (increases price)')
        red_patch = mpatches.Patch(color='red', label='Negative influence (decreases price)')
        ax.legend(handles=[green_patch, red_patch], loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to {save_path}")

        return fig

    def save_all_plots(self, df_dict, predictions, output_dir="charts"):
        """
        Save all available plots to disk

        Args:
            df_dict: Dictionary of stock DataFrames
            predictions: Dictionary of predictions
            output_dir: Output directory for charts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"💾 Saving charts to {output_dir}/...")

        for ticker in df_dict.keys():
            # Stock history
            fig = self.plot_stock_history(
                df_dict[ticker], ticker,
                save_path=os.path.join(output_dir, f"{ticker}_history.png")
            )
            if fig:
                plt.close(fig)

            # Prediction comparison
            if ticker in predictions and 'error' not in predictions[ticker]:
                fig = self.plot_prediction_comparison(
                    df_dict[ticker], predictions, ticker,
                    save_path=os.path.join(output_dir, f"{ticker}_prediction.png")
                )
                if fig:
                    plt.close(fig)

        # Overall performance
        fig = self.plot_model_performance(
            predictions,
            save_path=os.path.join(output_dir, "overall_performance.png")
        )
        if fig:
            plt.close(fig)

        print(f"✅ Charts saved successfully!")
