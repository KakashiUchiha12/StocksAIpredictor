#!/usr/bin/env python3
"""
AI Stock Predictor - Command Line Interface

A simple stock price prediction system using linear regression
with technical indicators and machine learning.

Usage:
    python main.py                    # Train and predict with default stocks
    python main.py --train-only      # Only train models
    python main.py --predict-only   # Only make predictions (models must exist)
"""

import argparse
import sys
from src.predictor import StockPredictionEngine
from src.config import DEFAULT_STOCKS


def print_banner():
    """Print the application banner"""
    print("""
🤖 AI STOCK PREDICTOR
═══════════════════════════════════════════════════════════════
An intelligent stock price prediction system using
machine learning and technical analysis.

Created with: Python, yfinance, ta, scikit-learn
═══════════════════════════════════════════════════════════════
""")


def print_model_performance(results):
    """Print training results in a nice format"""
    print("\n📊 MODEL TRAINING RESULTS")
    print("="*80)

    successful = 0
    for ticker, result in results.items():
        if result.get('success', False):
            successful += 1
            metrics = result.get('metrics', {})
            r2 = metrics.get('r2', 0)
            mae = metrics.get('mae', 0)
            directional_acc = metrics.get('directional_accuracy', 0)

            print(f"✅ {ticker:6}: R² {r2:6.3f} | MAE ${mae:6.2f} | Direction {directional_acc:5.1f}%")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ {ticker:6}: {error}")

    print(f"\n🏆 {successful}/{len(results)} models trained successfully")


def print_predictions(predictions):
    """Print predictions in a nice format"""
    print("\n🔮 STOCK PRICE PREDICTIONS")
    print("="*80)

    successful = 0
    for ticker, prediction in predictions.items():
        if 'error' not in prediction:
            successful += 1
            current_price = prediction.get('current_price', 0)
            predicted = prediction.get('predicted_price', 0)
            change_pct = prediction.get('price_change_pct', 0)
            direction = prediction.get('prediction_direction', '')
            confidence = prediction.get('confidence_interval')

            confidence_str = "+"             if confidence else "N/A"

            print(f"📈 {ticker:6}: ${current_price:7.2f} → ${predicted:7.2f} ({change_pct:+6.2f}%, {direction:6}) Confidence: {confidence_str}")
        else:
            error = prediction.get('error', 'Unknown error')
            print(f"❌ {ticker:6}: {error}")

    print(f"\n✅ Generated {successful}/{len(predictions)} predictions successfully")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='AI Stock Predictor - Predict stock prices using machine learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Train and predict all default stocks
  python main.py --stocks AAPL GOOGL       # Train and predict specific stocks
  python main.py --train-only            # Only train models
  python main.py --predict-only          # Only make predictions
  python main.py --days 365 --stocks TSLA  # Use 1 year data for Tesla
        """
    )

    parser.add_argument(
        '--stocks', '-s',
        nargs='*',
        default=DEFAULT_STOCKS,
        help=f'Stock tickers to analyze (default: {" ".join(DEFAULT_STOCKS)})'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=100,
        help='Historical data period in days (default: 100)'
    )

    parser.add_argument(
        '--prediction-days', '-p',
        type=int,
        default=1,
        help='Days to predict ahead (default: 1)'
    )

    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train models, do not make predictions'
    )

    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='Only make predictions, do not train (requires existing models)'
    )

    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh stock data (ignore cache)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with more details'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.train_only and args.predict_only:
        print("❌ Error: Cannot use both --train-only and --predict-only")
        sys.exit(1)

    if args.prediction_days < 1 or args.prediction_days > 30:
        print("❌ Error: Prediction days must be between 1 and 30")
        sys.exit(1)

    # Print banner
    print_banner()

    # Show configuration
    print(f"📈 Stocks: {', '.join(args.stocks)}")
    print(f"📅 Data Period: {args.days} days")
    print(f"🎯 Prediction Horizon: {args.prediction_days} day(s)")
    print(f"🔄 Force Refresh: {'Yes' if args.force_refresh else 'No'}")
    print()

    try:
        # Initialize the prediction engine
        engine = StockPredictionEngine()

        # Determine operation mode
        train = not args.predict_only
        predict = not args.train_only

        # Train models if requested
        if train:
            print("🧠 TRAINING PHASE")
            print("-" * 40)

            results = engine.train_all_models(
                tickers=args.stocks,
                days=args.days,
                force_refresh=args.force_refresh
            )

            if not args.verbose:
                print_model_performance(results)

        # Make predictions if requested
        if predict:
            print("🔮 PREDICTION PHASE")
            print("-" * 40)

            predictions = engine.predict_multiple_stocks(
                tickers=args.stocks,
                prediction_days=args.prediction_days
            )

            if not args.verbose:
                print_predictions(predictions)

        print(f"\n🎉 Stock prediction workflow completed!")

        # Show disclaimer
        print("\n⚠️  DISCLAIMER:")
        print("This is for educational purposes only. Past performance")
        print("does not guarantee future results. Not financial advice.")

    except KeyboardInterrupt:
        print("\n\n⚡ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
