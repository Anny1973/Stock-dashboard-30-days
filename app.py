import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# Your original functions remain unchanged
def compute_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_predictions(hist_data, future_prices, current_price, symbol):
    """Plot historical data and predictions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(hist_data.index[-200:], hist_data['Close'][-200:], label='Historical Close', linewidth=2)
    future_dates = pd.date_range(start=hist_data.index[-1] + timedelta(days=1),
                                periods=len(future_prices), freq='B')
    ax1.plot(future_dates, future_prices, label='Predicted', linewidth=2, linestyle='--')
    ax1.set_title(f'{symbol} - Stock Price Prediction (Next 30 Days)', fontsize=14)
    ax1.set_ylabel('Price (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist_data.index[-200:], hist_data['Volume'][-200:], label='Volume', color='orange', alpha=0.7)
    ax2.set_title('Trading Volume')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def predict_stock_price(symbol, period='2y', prediction_days=30):
    """
    Comprehensive stock price prediction for NSE/BSE stocks using LSTM.
    Includes OHLC analysis, PE ratio, and market factors.
    """

    # For NSE/BSE stocks, append .NS or .BO suffix
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        symbol = symbol + '.NS'  # Default to NSE

    st.write(f"Analyzing {symbol}...")

    # Step 1: Fetch historical OHLCV data
    stock = yf.Ticker(symbol)
    hist_data = stock.history(period=period)

    if hist_data.empty:
        st.error("No data found. Check symbol (use SYMBOL.NS for NSE, SYMBOL.BO for BSE)")
        return None

    # Step 2: Fetch fundamental data including PE ratio
    info = stock.info
    pe_ratio = info.get('trailingPE', 'N/A')
    forward_pe = info.get('forwardPE', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    sector = info.get('sector', 'N/A')

    st.subheader("Fundamental Analysis:")
    st.write(f"**Current PE Ratio:** {pe_ratio}")
    st.write(f"**Forward PE:** {forward_pe}")
    
    market_cap_display = market_cap
    if isinstance(market_cap, (int, float)):
        market_cap_display = f"{market_cap:,}"
    st.write(f"**Market Cap:** {market_cap_display}")
    
    st.write(f"**Sector:** {sector}")

    # Step 3: Technical Analysis - Add indicators
    hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
    hist_data['RSI'] = compute_rsi(hist_data['Close'])
    hist_data['Returns'] = hist_data['Close'].pct_change()

    # Volatility (important market factor)
    hist_data['Volatility'] = hist_data['Returns'].rolling(window=20).std()

    # Step 4: Prepare data for LSTM (using Close price + technical indicators)
    features = ['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility']
    data = hist_data[features].dropna()

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    X, y = [], []
    time_step = 60  # 60 days lookback

    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, 0])  # Predict Close price

    X, y = np.array(X), np.array(y)

    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 5: Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model (with progress bar)
    with st.spinner('Training LSTM model...'):
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    # Step 6: Make predictions for next 30 days
    last_60_days = scaled_data[-time_step:].copy() # Make a copy to avoid modifying original scaled_data
    future_predictions = []

    for _ in range(prediction_days):
        X_pred = np.array([last_60_days])
        predicted_scaled_close = model.predict(X_pred, verbose=0)[0, 0]
        future_predictions.append(predicted_scaled_close)

        # Create a new row of features for the next day, using predicted close and old values for other features
        new_day_features = last_60_days[-1].copy() # Start with the last day's features
        new_day_features[0] = predicted_scaled_close # Update the 'Close' price (index 0)

        # Append this new row (with correct dimensions) to the sequence
        last_60_days = np.append(last_60_days[1:], new_day_features.reshape(1, -1), axis=0)

    # Inverse transform predictions
    dummy = np.zeros((len(future_predictions), len(features)))
    dummy[:, 0] = np.array(future_predictions)
    future_prices = scaler.inverse_transform(dummy)[:, 0]

    current_price = hist_data['Close'][-1]

    # Step 7: Results and Analysis
    st.subheader("1-Month Price Prediction:")
    st.write(f"**Current Price:** ₹{current_price:.2f}")
    st.write(f"**Predicted Price (30 days):** ₹{future_prices[-1]:.2f}")
    change_pct = (future_prices[-1]/current_price - 1)*100
    st.write(f"**Expected Change:** {change_pct:+.1f}%")

    # Market factors summary
    st.subheader("Key Market Factors Considered:")
    st.write("- Historical OHLC patterns")
    st.write("- Moving averages (20/50 day)")
    rsi_val = hist_data['RSI'][-1]
    rsi_status = 'Oversold' if rsi_val < 30 else 'Overbought' if rsi_val > 70 else 'Neutral'
    st.write(f"- RSI: {rsi_val:.2f} ({rsi_status})")
    st.write(f"- Volatility: {hist_data['Volatility'][-1]*100:.1f}% (20-day)")
    st.write("- PE ratio valuation")
    st.write("- Trading volume trends")

    # Plot results
    fig = plot_predictions(hist_data, future_prices, current_price, symbol)
    st.pyplot(fig)

    return {
        'current_price': current_price,
        'predicted_price': future_prices[-1],
        'pe_ratio': pe_ratio,
        'prediction_change_pct': change_pct
    }

# Streamlit App
def main():
    st.title("Stock Price Prediction Dashboard")
    st.markdown("Predict stock prices using LSTM with technical and fundamental analysis.")
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        stock_symbol = st.text_input("Enter NSE/BSE stock symbol (e.g., RELIANCE, TCS):", value="RELIANCE").upper()
    with col2:
        period = st.selectbox("Historical Data Period:", ["1y", "2y", "5y"], index=1)
    
    if st.button("Predict Stock Price"):
        with st.spinner("Fetching data and running predictions..."):
            result = predict_stock_price(stock_symbol, period=period)
        
        if result:
            st.success(f"Analysis complete for {stock_symbol}")
            
            # Display key metrics
            st.subheader("Summary Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{result['current_price']:.2f}")
            with col2:
                st.metric("Predicted Price", f"₹{result['predicted_price']:.2f}")
            with col3:
                st.metric("Expected Change", f"{result['prediction_change_pct']:+.1f}%")

if __name__ == "__main__":
    main()