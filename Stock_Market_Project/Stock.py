import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import io
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Stock Market Prediction", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Market Prediction using LSTM")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model parameters
    st.subheader("Model Parameters")
    lookback_days = st.slider("Lookback Period (days)", 30, 120, 60, 10)
    future_days = st.slider("Future Prediction Days", 1, 30, 7, 1)
    epochs = st.slider("Training Epochs", 10, 100, 50, 10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    st.subheader("LSTM Architecture")
    lstm_units_1 = st.slider("LSTM Layer 1 Units", 32, 128, 50, 16)
    lstm_units_2 = st.slider("LSTM Layer 2 Units", 32, 128, 50, 16)
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.1)
    
    train_button = st.button("🚀 Train Model", use_container_width=True)

# Load data
@st.cache_data
def load_data():
    # Sample data - replace with uploaded file data
    data = """Date,Open,High,Low,Close,Shares Traded,Turnover (₹ Cr)
05-JAN-2026,26333.7,26373.2,26210.05,26250.3,338777649,25742.83
02-JAN-2026,26155.1,26340,26118.4,26328.55,357547806,23770.13
01-JAN-2026,26173.3,26197.55,26113.4,26146.55,425631910,23454.66
31-DEC-2025,25971.05,26187.95,25969,26129.6,246314941,20703.99
30-DEC-2025,25940.9,25976.75,25878,25938.85,396893959,39492.31"""
    
    df = pd.read_csv(io.StringIO(data))
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def process_dataframe(df):
    """Process and validate the dataframe"""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Find the date column (case-insensitive)
    date_col = None
    for col in df.columns:
        if col.lower() == 'date':
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("❌ No 'Date' column found in the CSV file. Available columns: " + ", ".join(df.columns))
    
    # Rename to standard 'Date' if different
    if date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})
    
    # Try different date formats
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    except:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            except:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                except:
                    raise ValueError("❌ Unable to parse dates. Please ensure your Date column is in format: DD-MMM-YYYY (e.g., 05-JAN-2026)")
    
    # Check for required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Try case-insensitive matching
        for col in missing_cols[:]:
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    df = df.rename(columns={df_col: col})
                    missing_cols.remove(col)
                    break
    
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {', '.join(missing_cols)}")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

# File uploader
uploaded_file = st.file_uploader("Upload your stock data (CSV file)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = process_dataframe(df)
        st.success(f"✅ Data loaded successfully! {len(df)} rows found.")
    except Exception as e:
        st.error(str(e))
        st.info("📋 Expected CSV format:\n- Date column (DD-MMM-YYYY format, e.g., 05-JAN-2026)\n- Open, High, Low, Close columns\n- Example: Date,Open,High,Low,Close")
        df = load_data()
        st.info("Using sample data instead. Please upload a valid CSV file.")
else:
    df = load_data()
    st.info("👆 Upload your CSV file or use the sample data to get started")

# Display data overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Days", len(df))
with col2:
    st.metric("Latest Close", f"₹{df['Close'].iloc[-1]:,.2f}")
with col3:
    change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    pct_change = (change / df['Close'].iloc[-2]) * 100
    st.metric("Day Change", f"₹{change:,.2f}", f"{pct_change:.2f}%")
with col4:
    st.metric("Avg Volume", f"{df['Shares Traded'].mean()/1e7:.2f}Cr")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔮 Predictions", "📈 Performance"])

with tab1:
    st.subheader("Historical Stock Data")
    
    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    )])
    
    fig.update_layout(
        title='Stock Price Movement (Candlestick Chart)',
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    fig_vol = go.Figure(data=[go.Bar(
        x=df['Date'],
        y=df['Shares Traded'],
        name='Volume',
        marker_color='rgba(50, 171, 96, 0.6)'
    )])
    
    fig_vol.update_layout(
        title='Trading Volume',
        yaxis_title='Shares Traded',
        xaxis_title='Date',
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Data table
    st.subheader("Recent Data")
    st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    st.subheader("LSTM Model Training")
    
    if train_button:
        with st.spinner("🔄 Preparing data and training model..."):
            # Prepare data
            data = df[['Close']].values
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            def create_sequences(data, lookback):
                X, y = [], []
                for i in range(lookback, len(data)):
                    X.append(data[i-lookback:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)
            
            X, y = create_sequences(scaled_data, lookback_days)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Train-test split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(lstm_units_1, return_sequences=True, input_shape=(lookback_days, 1)),
                Dropout(dropout_rate),
                LSTM(lstm_units_2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback for progress
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.6f} - MAE: {logs['mae']:.6f}")
            
            # Train model
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                callbacks=[early_stop, StreamlitCallback()],
                verbose=0
            )
            
            progress_bar.empty()
            status_text.empty()
            
            # Make predictions
            train_predict = model.predict(X_train, verbose=0)
            test_predict = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
            train_mae = mean_absolute_error(y_train_actual, train_predict)
            test_mae = mean_absolute_error(y_test_actual, test_predict)
            train_r2 = r2_score(y_train_actual, train_predict)
            test_r2 = r2_score(y_test_actual, test_predict)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['history'] = history.history
            st.session_state['metrics'] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            st.session_state['predictions'] = {
                'train': train_predict,
                'test': test_predict,
                'y_train': y_train_actual,
                'y_test': y_test_actual,
                'split': split
            }
            
            st.success("✅ Model trained successfully!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test RMSE", f"₹{test_rmse:.2f}")
            with col2:
                st.metric("Test MAE", f"₹{test_mae:.2f}")
            with col3:
                st.metric("Test R² Score", f"{test_r2:.4f}")
    
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Train Model' to start")
        
        # Show model architecture preview
        st.subheader("Model Architecture Preview")
        st.code(f"""
Sequential Model:
├── LSTM Layer 1: {lstm_units_1} units (return_sequences=True)
├── Dropout: {dropout_rate}
├── LSTM Layer 2: {lstm_units_2} units
├── Dropout: {dropout_rate}
├── Dense Layer: 25 units
└── Output Layer: 1 unit

Training Configuration:
├── Lookback Period: {lookback_days} days
├── Batch Size: {batch_size}
├── Epochs: {epochs}
└── Optimizer: Adam
        """, language='text')

with tab3:
    st.subheader("Future Price Predictions")
    
    if 'model' in st.session_state:
        try:
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            
            # Prepare last sequence for prediction
            last_sequence = df['Close'].values[-lookback_days:]
            last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
            
            # Predict future
            future_predictions = []
            current_sequence = last_sequence_scaled.copy()
            
            for _ in range(future_days):
                current_sequence_reshaped = current_sequence.reshape(1, lookback_days, 1)
                next_pred = model.predict(current_sequence_reshaped, verbose=0)
                future_predictions.append(next_pred[0, 0])
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            # Inverse transform and flatten
            future_predictions_array = np.array(future_predictions).reshape(-1, 1)
            future_predictions_scaled = scaler.inverse_transform(future_predictions_array)
            future_predictions_flat = future_predictions_scaled.flatten()
            
            # Create future dates (business days only - excluding weekends)
            last_date = df['Date'].iloc[-1]
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)
            
            # Ensure arrays are same length
            min_length = min(len(future_dates), len(future_predictions_flat))
            future_dates = future_dates[:min_length]
            future_predictions_flat = future_predictions_flat[:min_length]
            
            
            fig = go.Figure()
            
            
            fig.add_trace(go.Scatter(
                x=df['Date'][-100:],
                y=df['Close'][-100:],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions_flat,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f'Stock Price Prediction - Next {min_length} Trading Days',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction table
            st.subheader("Predicted Prices")
            
            # Calculate changes
            last_close = df['Close'].iloc[-1]
            changes = future_predictions_flat - last_close
            change_pct = (changes / last_close) * 100
            
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_predictions_flat,
                'Change from Last': changes,
                'Change %': change_pct
            })
            
            st.dataframe(pred_df.style.format({
                'Predicted Price': '₹{:.2f}',
                'Change from Last': '₹{:.2f}',
                'Change %': '{:.2f}%'
            }), use_container_width=True)
            
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{last_close:,.2f}")
            with col2:
                predicted_last = future_predictions_flat[-1]
                st.metric(f"Price in {min_length} Days", f"₹{predicted_last:,.2f}")
            with col3:
                total_change = ((predicted_last - last_close) / last_close) * 100
                st.metric("Expected Change", f"{total_change:.2f}%")
        
        except Exception as e:
            st.error(f"❌ Error generating predictions: {str(e)}")
            st.info("Please try retraining the model with different parameters.")
            
    else:
        st.warning("⚠️ Please train the model first in the 'Model Training' tab")

with tab4:
    st.subheader("Model Performance Analysis")
    
    if 'history' in st.session_state:
        history = st.session_state['history']
        metrics = st.session_state['metrics']
        predictions = st.session_state['predictions']
        
        # Training history
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Mean Absolute Error')
        )
        
        fig.add_trace(
            go.Scatter(y=history['loss'], name='Train Loss', mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=history['mae'], name='Train MAE', mode='lines'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_mae'], name='Val MAE', mode='lines'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Set**")
            st.metric("RMSE", f"₹{metrics['train_rmse']:.2f}")
            st.metric("MAE", f"₹{metrics['train_mae']:.2f}")
            st.metric("R² Score", f"{metrics['train_r2']:.4f}")
        
        with col2:
            st.markdown("**Test Set**")
            st.metric("RMSE", f"₹{metrics['test_rmse']:.2f}")
            st.metric("MAE", f"₹{metrics['test_mae']:.2f}")
            st.metric("R² Score", f"{metrics['test_r2']:.4f}")
        
        # Prediction vs Actual
        st.subheader("Predictions vs Actual Prices")
        
        # Prepare data for visualization
        train_dates = df['Date'][lookback_days:lookback_days+len(predictions['train'])]
        test_dates = df['Date'][lookback_days+len(predictions['train']):lookback_days+len(predictions['train'])+len(predictions['test'])]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=predictions['y_train'].flatten(),
            mode='lines',
            name='Actual (Train)',
            line=dict(color='blue', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=predictions['train'].flatten(),
            mode='lines',
            name='Predicted (Train)',
            line=dict(color='lightblue', width=1.5, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions['y_test'].flatten(),
            mode='lines',
            name='Actual (Test)',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions['test'].flatten(),
            mode='lines',
            name='Predicted (Test)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Model Predictions vs Actual Prices',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ Please train the model first to see performance metrics")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Stock Market Prediction using LSTM Deep Learning Model</p>
        <p>⚠️ Disclaimer: This is for educational purposes only. Do not make investment decisions based solely on these predictions.</p>
    </div>
""", unsafe_allow_html=True)