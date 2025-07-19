import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import plotly.graph_objects as go
import warnings
from datetime import datetime
import pytz

warnings.filterwarnings("ignore")
np.random.seed(42)

MODEL_FEATURES = [
    'PE Ratio', 'EPS', 'ROE', 'ROA', 'Debt/Equity',
    'Dividend Yield (%)', 'Revenue Growth', 'Profit Margin', 'EV/EBITDA', 'Beta'
]

def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    weekday = now.weekday()
    market_start = datetime.strptime("09:15", "%H:%M").time()
    market_end = datetime.strptime("15:30", "%H:%M").time()
    is_open = market_start <= current_time <= market_end and weekday < 5
    return is_open, now.strftime('%A, %d %B %Y %I:%M:%S %p'), market_start, market_end

def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return pd.Series({
        'PE Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'ROE': info.get('returnOnEquity'),
        'ROA': info.get('returnOnAssets'),
        'Debt/Equity': info.get('debtToEquity'),
        'Dividend Yield (%)': (info.get('dividendYield') or 0) * 100,
        'Revenue Growth': info.get('revenueGrowth'),
        'Profit Margin': info.get('profitMargins'),
        'Beta': info.get('beta'),
        'EV/EBITDA': info.get('enterpriseToEbitda')
    })

def get_technical_indicators(df):
    # Always convert to Series (1D)
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()
    return pd.Series({
        'RSI': float(RSIIndicator(close).rsi().iloc[-1]),
        'MACD': float(MACD(close).macd().iloc[-1]),
        'MACD Signal': float(MACD(close).macd_signal().iloc[-1]),
        'Stochastic': float(StochasticOscillator(high, low, close).stoch().iloc[-1]),
        'ADX': float(ADXIndicator(high, low, close).adx().iloc[-1]),
        'OBV': float(OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]),
        'SMA20': float(SMAIndicator(close, 20).sma_indicator().iloc[-1]),
        'BB High': float(BollingerBands(close).bollinger_hband().iloc[-1]),
        'BB Low': float(BollingerBands(close).bollinger_lband().iloc[-1]),
        'Current Price': float(close.iloc[-1])
    })

def train_ml_model():
    df = pd.DataFrame(np.random.rand(100, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
    df['Target'] = np.random.rand(100) * 100
    X = df[MODEL_FEATURES]
    y = df['Target']
    rf = RandomForestRegressor().fit(X, y)
    gb = GradientBoostingRegressor().fit(X, y)
    return rf, gb

def predict_score(models, input_data):
    input_df = pd.DataFrame([input_data])[MODEL_FEATURES].fillna(0)
    rf_score = models[0].predict(input_df)[0]
    gb_score = models[1].predict(input_df)[0]
    return round((rf_score + gb_score) / 2, 2)

def prepare_ann_data(df):
    close = df['Close'].squeeze()
    data = pd.DataFrame({
        'RSI': RSIIndicator(close).rsi(),
        'MACD': MACD(close).macd(),
        'Stochastic': StochasticOscillator(df['High'].squeeze(), df['Low'].squeeze(), close).stoch(),
        'ADX': ADXIndicator(df['High'].squeeze(), df['Low'].squeeze(), close).adx(),
        'OBV': OnBalanceVolumeIndicator(close, df['Volume'].squeeze()).on_balance_volume(),
        'Return': close.pct_change(),
        'Direction': np.where(close.shift(-1) > close, 1, 0)
    }).dropna()
    X = data.drop('Direction', axis=1).values
    y = data['Direction'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def build_and_train_ann(X, y):
    model = keras.models.Sequential([
        keras.layers.Dense(16, activation='relu', input_dim=X.shape[1]),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

def ann_predict_next_direction(model, scaler, latest_features):
    X_latest = scaler.transform([latest_features])
    prob = model.predict(X_latest)[0][0]
    return "UP" if prob > 0.5 else "DOWN", prob

def generate_signals(tech):
    signals = []
    now = tech['Current Price']
    bb_low = tech['BB Low']
    bb_high = tech['BB High']
    buy, sell = False, False

    if tech['RSI'] < 30:
        signals.append(f"📉 RSI = {tech['RSI']:.2f} → Oversold (BUY)")
        buy = True
    if tech['MACD'] > tech['MACD Signal']:
        signals.append("🟢 MACD Bullish Crossover → BUY")
        buy = True
    if tech['Stochastic'] < 20:
        signals.append(f"📈 Stochastic = {tech['Stochastic']:.2f} → Buy Zone")
        buy = True

    if tech['RSI'] > 70:
        signals.append(f"📈 RSI = {tech['RSI']:.2f} → Overbought (SELL)")
        sell = True
    if tech['MACD'] < tech['MACD Signal']:
        signals.append("🔻 MACD Bearish Crossover → SELL")
        sell = True
    if tech['Stochastic'] > 80:
        signals.append(f"📉 Stochastic = {tech['Stochastic']:.2f} → Sell Zone")
        sell = True

    signals.append("------")

    if buy:
        signals.append(f"✅ Suggested BUY Price: ₹{bb_low:.2f}")
        signals.append(f"🎯 Target After Buy: ₹{bb_high:.2f}")

    if sell:
        signals.append(f"🛑 Suggested SELL Price: ₹{bb_high:.2f}")
        signals.append(f"📉 Watch Drop Target: ₹{bb_low:.2f}")

    if not buy and not sell:
        signals.append("⚠️ No strong Buy/Sell signal currently.")
        signals.append(f"🔎 Range: ₹{bb_low:.2f} - ₹{bb_high:.2f} (Bollinger Bands)")

    return signals

def main():
    st.set_page_config(page_title="📊 AI Stock Analyzer", layout="wide")
    st.title("🧠 AI-Powered Indian Stock Analyzer (with ANN Direction Forecast)")

    open_status, curr_time, start, end = is_market_open()
    st.markdown(f"🕒 **Current Time (IST):** {curr_time}")
    st.markdown(f"🕘 **NSE Market Hours (Mon–Fri):** {start} – {end}")
    if open_status:
        st.success("✅ **Market is OPEN**")
    else:
        st.warning("❌ **Market is CLOSED**")

    stock = st.text_input("🔍 Enter NSE Stock Code (e.g., RELIANCE.NS, SUZLON.NS, TCS.NS):", "SUZLON.NS")
    if not stock.strip():
        st.info("Please enter a stock like INFY.NS, TCS.NS")
        return

    try:
        df = yf.download(stock.strip(), period="1y", interval="1d", progress=False)
        if df.empty:
            st.error("❌ No data found for this ticker.")
            return

        fund = get_fundamental_data(stock)
        tech = get_technical_indicators(df)
        models = train_ml_model()
        score = predict_score(models, fund)

        X_ann, y_ann, scaler_ann = prepare_ann_data(df)
        ann_model = build_and_train_ann(X_ann, y_ann)
        last_features = X_ann[-1]
        next_direction, probability = ann_predict_next_direction(ann_model, scaler_ann, last_features)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📘 Fundamentals")
            st.dataframe(fund.to_frame(name="Value"))
        with col2:
            st.subheader("📗 Technical Indicators")
            st.dataframe(tech.to_frame(name="Value"))

        st.subheader("📉 Price Chart with 20-day SMA")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].squeeze(), name="Close"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean().squeeze(), name="SMA 20"))
        fig.update_layout(height=400, margin=dict(t=20, b=20), legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"🤖 AI Stock Score (0–100): **{score}**")
        st.info(f"🧠 ANN Prediction: **{next_direction}** ({probability*100:.1f}% confidence for UP)")

        st.subheader("📌 Buy/Sell Signals")
        for signal in generate_signals(tech):
            st.markdown(f"- {signal}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
