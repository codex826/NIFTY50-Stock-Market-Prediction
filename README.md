# 📈 NIFTY 50 Stock Market Prediction System

A comprehensive Machine Learning and Deep Learning application for predicting NIFTY 50 stock prices using LSTM (Long Short-Term Memory) neural networks with an interactive Streamlit frontend.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Overview

This project implements a stock market prediction system that leverages LSTM neural networks to analyze historical NIFTY 50 data and forecast future stock prices. The application provides an intuitive web interface for data upload, model training, visualization, and price prediction.

## ✨ Features

### 📊 Data Analysis & Visualization
- **CSV Data Upload**: Easy drag-and-drop file upload
- **Interactive Charts**: Line charts, candlestick patterns, and trend analysis
- **Statistical Summary**: Mean, median, standard deviation, variance, and range
- **Multi-column Support**: Analyze Open, High, Low, Close prices

### 🤖 Machine Learning Capabilities
- **LSTM Neural Network**: 3-layer deep learning architecture
- **Customizable Parameters**: Adjust lookback period, epochs, and batch size
- **Dropout Regularization**: Prevents overfitting (20% dropout rate)
- **Train-Test Split**: 80-20 ratio with validation

### 📈 Prediction & Forecasting
- **Historical Predictions**: Compare actual vs predicted values
- **Future Forecasting**: Predict 7-60 days ahead
- **Performance Metrics**: RMSE, MAE, and R² scores
- **Visual Forecasts**: Interactive charts with confidence indicators

### 🎨 User Interface
- **Responsive Design**: Clean, modern interface
- **Real-time Progress**: Live training updates
- **Interactive Plots**: Zoom, pan, and hover tooltips (Plotly)
- **Tabbed Navigation**: Organized information display

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| **Frontend** | Streamlit |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Algorithm** | LSTM (Recurrent Neural Network) |

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- GPU support optional (for faster training)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/nifty50-prediction.git
cd nifty50-prediction
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create requirements.txt
```text
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
tensorflow==2.13.0
plotly==5.16.1
```

## 💻 Usage

### Running the Application

1. **Start the Streamlit App**
```bash
streamlit run app.py
```

2. **Access the Application**
   - Open your browser
   - Navigate to `http://localhost:8501`

### Step-by-Step Guide

#### Step 1: Upload Data
- Click "Browse files" in the sidebar
- Upload your NIFTY 50 CSV file
- Supported format: Date, Open, High, Low, Close, Volume

#### Step 2: Configure Parameters
- **Lookback Period**: 30-120 days (default: 60)
- **Forecast Days**: 7-60 days (default: 30)
- **Training Epochs**: 20-100 (default: 50)
- **Batch Size**: 16, 32, or 64 (default: 32)

#### Step 3: Select Price Column
- Choose which price to predict (Close, Open, High, Low)
- View data overview and statistics

#### Step 4: Train Model
- Click "Train Model" button
- Monitor training progress
- View performance metrics

#### Step 5: Analyze Results
- Examine actual vs predicted charts
- Review forecast predictions
- Export results (optional)

## 📊 CSV Data Format

### Required Format
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,21500.50,21650.75,21450.25,21580.00,1500000
2024-01-02,21590.00,21700.50,21500.00,21620.25,1600000
2024-01-03,21630.75,21750.00,21580.50,21690.00,1700000
```

### Column Requirements
- **Date**: Any format (YYYY-MM-DD, DD/MM/YYYY, etc.)
- **Price Columns**: Open, High, Low, Close (at least one required)
- **Volume**: Optional but recommended
- **Order**: Chronological (automatically sorted)

### Sample Data
You can download sample NIFTY 50 data from:
- [NSE India](https://www.nseindia.com/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Google Finance](https://www.google.com/finance/)

## 🧠 Model Architecture

```
Input Layer (Lookback days, 1 feature)
    ↓
LSTM Layer 1 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (1 unit)
    ↓
Output (Predicted Price)
```

### Hyperparameters
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Activation**: Tanh (LSTM default)
- **Dropout Rate**: 0.2 (20%)

## 📈 Performance Metrics

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Coefficient of determination (0-1, higher is better)

### Interpretation
- **R² > 0.8**: Excellent fit
- **R² 0.6-0.8**: Good fit
- **R² < 0.6**: May need tuning
- **RMSE/MAE**: Lower is better (in price units)

## 🎯 Use Cases

1. **Day Traders**: Short-term price predictions
2. **Swing Traders**: Medium-term trend analysis
3. **Long-term Investors**: Market trend identification
4. **Financial Analysts**: Technical analysis support
5. **Researchers**: Time-series forecasting studies
6. **Students**: Learn ML/DL applications in finance

## ⚙️ Configuration Options

### Sidebar Controls
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Lookback Period | 30-120 | 60 | Days of history to use |
| Forecast Days | 7-60 | 30 | Future prediction period |
| Training Epochs | 20-100 | 50 | Training iterations |
| Batch Size | 16/32/64 | 32 | Training batch size |

## 📸 Screenshots

### Main Dashboard
- Data overview with key metrics
- Interactive price charts
- Statistical analysis

### Model Training
- Real-time progress tracking
- Loss curve visualization
- Performance metrics display

### Predictions
- Historical predictions comparison
- Future forecast visualization
- Detailed prediction tables

## 🔧 Troubleshooting

### Common Issues

**Issue**: Model training is slow
- **Solution**: Reduce epochs or use GPU acceleration

**Issue**: CSV file not loading
- **Solution**: Check file format, ensure Date column exists

**Issue**: Poor prediction accuracy
- **Solution**: Increase lookback period, add more training data

**Issue**: Memory error during training
- **Solution**: Reduce batch size or lookback period

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Images
![image alt](https://github.com/codex826/NIFTY50-Stock-Market-Prediction/blob/f75c4a0fe08feb0dbe64c0a340f2b0e5980e69e7/Screenshot%202026-01-06%20124420.png)

![image alt](https://github.com/codex826/NIFTY50-Stock-Market-Prediction/blob/7c0f5851d251379e992fcbccdec89f8fe28641a2/Screenshot%202026-01-06%20124339.png)

![image alt](https://github.com/codex826/NIFTY50-Stock-Market-Prediction/blob/8702f28c5e6214d4072a594e3cee105e0d7d7b0a/Screenshot%202026-01-06%20124402.png)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sushant Nichat** - *Initial work* - [GitHub](https://github.com/codex826)

## 🙏 Acknowledgments

- NIFTY 50 data providers (NSE India, Yahoo Finance)
- TensorFlow and Keras teams
- Streamlit community
- Open-source contributors

## 📧 Contact

- **Email**: sushantnichat@gmail.com

## 🔮 Future Enhancements

- [ ] Multi-stock support
- [ ] Sentiment analysis integration
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Model comparison (GRU, Transformer, Prophet)
- [ ] Real-time data fetching via API
- [ ] Portfolio optimization
- [ ] Alert notifications
- [ ] Model export/import functionality
- [ ] Backtesting capabilities
- [ ] Mobile app version

## 📚 References

- [LSTM Networks](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Stock Market Analysis](https://www.investopedia.com/)

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain. Do not use this application as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions.

---

**Made with ❤️ using Python, TensorFlow, and Streamlit**


**⭐ Star this repository if you found it helpful!**




