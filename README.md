# 🤖 AI Stock Predictor

An intelligent stock price prediction system using machine learning and technical analysis. Built with Python, featuring both command-line and web interfaces.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-Linear%20Regression-orange.svg)

## 🚀 Features

- **🧠 Machine Learning**: Linear regression with technical indicators
- **📊 Real-time Data**: Yahoo Finance integration with automatic caching
- **🎯 Multiple Predictions**: 1-30 day prediction horizons
- **📈 Interactive Charts**: Beautiful visualizations with Plotly
- **💾 Model Persistence**: Automatic model saving and loading
- **🌐 Web Interface**: Streamlit-powered dashboard
- **⚡ Command Line**: Full CLI support for automation
- **🎨 Advanced Analytics**: Feature importance and performance metrics

## 📋 Table of Contents

- [Quick Start](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
  - [Web App](#web-app)
  - [Command Line](#command-line)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Screenshots](#screenshots)
- [Disclaimer](#disclaimer)

## <a name="quickstart"></a>⚡ Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Launch web app**:
```bash
streamlit run app.py
```

3. **Or use command line**:
```bash
python main.py
```

## <a name="installation"></a>📦 Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection (for stock data downloads)

### Step-by-step Installation

1. **Clone or download the project**:
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import yfinance, pandas_ta, sklearn; print('✅ All dependencies installed')"
```

## <a name="usage"></a>🎯 Usage

### Web App Interface

The easiest way to use the stock predictor:

```bash
streamlit run app.py
```

This launches a beautiful web interface at `http://localhost:8501`

**Web App Features**:
- 📊 Interactive dashboard
- 🎛️ Easy configuration
- 📈 Real-time charts
- 🎯 One-click predictions
- 💾 Automatic model management

### Command Line Interface

For automation and scripting:

```bash
# Basic usage with default stocks
python main.py

# Train models only
python main.py --train-only

# Make predictions only (requires trained models)
python main.py --predict-only --stocks AAPL GOOGL

# Custom configuration
python main.py --stocks TSLA NVDA --days 730 --prediction-days 7 --verbose

# Force fresh data download
python main.py --force-refresh
```

**CLI Options**:
```
usage: main.py [-h] [--stocks [STOCKS [STOCKS ...]]] [--days DAYS]
               [--prediction-days PREDICTION_DAYS] [--train-only]
               [--predict-only] [--force-refresh] [--verbose]

optional arguments:
  --stocks STOCKS [STOCKS ...]  Stock tickers (default: AAPL GOOGL MSFT TSLA AMZN)
  --days DAYS                  Historical data period in days (default: 500)
  --prediction-days PREDICTION_DAYS
                              Days to predict ahead (default: 1)
  --train-only                Only train models, do not make predictions
  --predict-only              Only make predictions (requires existing models)
  --force-refresh             Force refresh stock data (ignore cache)
  --verbose, -v               Verbose output with more details
```

### Python API Usage

```python
from src.predictor import StockPredictionEngine

# Initialize the engine
engine = StockPredictionEngine()

# Train models for stocks
training_results = engine.train_all_models(['AAPL', 'GOOGL', 'MSFT'])

# Make predictions
predictions = engine.predict_multiple_stocks(['AAPL', 'GOOGL'])

# Get predictions for specific stock
apple_pred = engine.predict_stock('AAPL', prediction_days=5)

# Save/load models
engine.train_single_model('TSLA')  # Model is automatically saved
tsla_pred = engine.predict_stock('TSLA')  # Model is loaded from disk
```

## <a name="technical-details"></a>🛠️ Technical Details

### Machine Learning Model
- **Algorithm**: Multiple Linear Regression with feature scaling
- **Features**: 20+ technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
- **Training**: 80/20 train/test split with cross-validation
- **Metrics**: R², MAE, MAPE, Directional Accuracy

### Data Pipeline
1. **Data Collection**: Yahoo Finance API (cached for 30 days)
2. **Feature Engineering**: Technical indicators via pandas-ta
3. **Preprocessing**: Standard scaling and NaN handling
4. **Training**: Cross-validated linear regression
5. **Inference**: Real-time predictions with confidence intervals

### Supported Technical Indicators
- **Moving Averages**: SMA (10, 20, 50), EMA (12, 26)
- **Momentum**: RSI (14), MACD, Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR
- **Volume**: On Balance Volume, Volume Ratios
- **Custom**: Price momentum, Lag features, Rolling statistics

### Model Persistence
- **Format**: Joblib (pickle-based)
- **Location**: `./models/` directory
- **Naming**: `{ticker}_model.pkl`
- **Content**: Model, scaler, metrics, feature names

## <a name="configuration"></a>⚙️ Configuration

Edit `src/config.py` to customize behavior:

```python
# Default stocks
DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# Training settings
TRAINING_DAYS = 500          # Historical data period
TEST_SPLIT = 0.2             # Train/test split ratio
RANDOM_STATE = 42            # Reproducibility

# Model settings
MODEL_ACCURACY_THRESHOLD = 0.7  # Minimum R² to save model
```

## <a name="screenshots"></a>📸 Screenshots

### Web Interface Dashboard
*(Beautiful Streamlit interface with metrics, charts, and predictions)*

### Prediction Results
*(Clear display of current vs predicted prices with confidence indicators)*

### Performance Analytics
*(Interactive charts showing model accuracy and comparative analysis)*

## 🐛 Troubleshooting

### Common Issues

**1. "Module not found" errors**:
```bash
pip install -r requirements.txt
```

**2. No data downloaded**:
```bash
# Clear cache and retry
rm -rf data/ models/
python main.py --force-refresh
```

**3. Model training failures**:
- Check internet connection
- Try different stocks
- Reduce training period: `python main.py --days 365`

**4. Streamlit not starting**:
```bash
pip install --upgrade streamlit
streamlit run app.py
```

### Performance Tips

- **Use caching**: Avoid `--force-refresh` for repeated runs
- **Reduce training data**: Use `--days 365` for faster training
- **Select fewer stocks**: Process 2-3 stocks at a time initially
- **Monitor memory**: Close browser tabs if memory issues occur

### Data Sources

- **Primary**: Yahoo Finance (yfinance library)
- **Cache Location**: `./data/` directory
- **Rate Limits**: 2,000 requests/hour (handles automatically)
- **Data Types**: OHLCV (Open, High, Low, Close, Volume)

## 📁 Project Structure

```
stock-predictor/
├── src/                          # Core application
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── data_loader.py           # Yahoo Finance interface
│   ├── feature_engineer.py      # Technical indicators
│   ├── model_trainer.py         # ML model training
│   ├── predictor.py             # Prediction orchestration
│   └── visualizer.py            # Charts and plotting
├── models/                       # Saved ML models
├── data/                         # Cached stock data
├── charts/                       # Generated visualizations
├── app.py                        # Streamlit web interface
├── main.py                       # Command-line interface
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Adding New Stocks
```python
# Add to DEFAULT_STOCKS in config.py
DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
```

### Adding New Indicators
```python
# In feature_engineer.py
def add_custom_indicators(self):
    # Add your indicator logic here
    self.df['my_custom_indicator'] = calculate_custom_indicator(self.df['Close'])
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **Libraries**: yfinance, pandas-ta, scikit-learn, Streamlit
- **Data Provider**: Yahoo Finance
- **Inspiration**: Quantitative finance and algorithmic trading communities

## <a name="disclaimer"></a>⚠️ Important Disclaimer

**This application is for educational and research purposes only.**

- 📊 **Not Financial Advice**: Predictions are based on historical data using machine learning. Past performance does not guarantee future results.
- 🏦 **Professional Consultation Required**: This tool should not replace professional financial advice or investment decisions.
- 💰 **Investment Risk**: All investments carry risk. Only invest money you can afford to lose.
- 🤖 **Model Limitations**: AI predictions have uncertainty and may not account for breaking news, geopolitical events, economic factors, or market sentiment.
- 🔄 **No Guarantees**: Results may vary significantly, and there are no guarantees of accuracy, profitability, or reliability.

**By using this application, you acknowledge that you understand and accept these limitations.**

---

**Created with ❤️ for the quantitative finance community**

*If you find this project helpful, please ⭐ star the repository!*
