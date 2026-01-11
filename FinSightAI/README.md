# FinSight AI – Algorithmic Stock Trend Forecaster

A complete machine learning system for predicting stock price trends using Linear Regression (baseline) and LSTM (primary model). This project implements financial feature engineering, time-series modeling, and an interactive Streamlit dashboard.

## 🎯 Features

- **Financial Feature Engineering**: RSI (14-period), MACD (12-26-9), 50-Day SMA
- **Dual Models**: Linear Regression (baseline) and LSTM (TensorFlow/Keras)
- **7-Day Forecast**: Predicts closing price 7 days ahead
- **Performance Metrics**: RMSE comparison with 15% improvement target
- **Interactive Dashboard**: Streamlit app with Plotly visualizations
- **Backtesting**: Rolling window predictions across test set

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## 🚀 Installation

1. Navigate to the FinSightAI directory:
   ```bash
   cd FinSightAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Enter a valid S&P 500 stock ticker (e.g., AAPL, MSFT, GOOGL)
   - Click "🚀 Train Models" button
   - Wait for training to complete (LSTM training may take a few minutes)
   - View results, visualizations, and metrics

## 📊 Project Structure

```
FinSightAI/
│
├── data/
│   └── fetch_data.py          # Data fetching and preprocessing
├── features/
│   └── indicators.py          # Financial feature engineering (RSI, MACD, SMA)
├── models/
│   ├── linear_model.py        # Linear Regression model
│   └── lstm_model.py          # LSTM model (TensorFlow/Keras)
├── evaluation/
│   └── metrics.py             # Evaluation metrics (RMSE, etc.)
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔬 Technical Details

### Data Pipeline
- Fetches 5+ years of historical stock data using yfinance
- Handles missing values with forward/backward fill
- Normalizes features using MinMaxScaler
- Creates rolling windows for LSTM (60-day lookback)
- Creates 7-day ahead labels

### Feature Engineering
- **RSI (14-period)**: Relative Strength Index
- **MACD (12-26-9)**: Moving Average Convergence Divergence
- **50-Day SMA**: Simple Moving Average

### Models
- **Linear Regression**: Baseline model using Scikit-learn
- **LSTM**: Primary model with:
  - Input shape: (60, features)
  - Architecture: LSTM → Dropout → Dense
  - Loss: MSE
  - Optimizer: Adam

### Evaluation
- Train/Test split: 80/20
- Metric: RMSE (Root Mean Squared Error)
- Target: LSTM RMSE must be at least 15% lower than Linear Regression RMSE

## 📈 Dashboard Features

- Historical price chart
- Predicted vs Actual prices comparison
- RMSE comparison (Linear vs LSTM)
- Improvement percentage
- Backtesting results with rolling windows
- Interactive Plotly charts

## ⚠️ Notes

- LSTM training may take several minutes depending on your hardware
- The system attempts to achieve 15% improvement through iterative hyperparameter tuning
- Results are printed to console/log for verification
- Internet connection required for fetching stock data

## 📝 License

This project is provided as-is for educational and demonstration purposes.
