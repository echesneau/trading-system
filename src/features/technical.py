# src/features/technical.py
import pandas as pd
import numpy as np
import ta  # Bibliothèque d'analyse technique


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques pour les données de marché.

    Args:
        data: DataFrame contenant les colonnes OHLCV (Open, High, Low, Close, Volume)

    Returns:
        DataFrame original enrichi avec les indicateurs techniques
    """
    df = data.copy()

    # Vérification des colonnes requises
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Données manquantes pour calculer les indicateurs: {missing}")

    # 1. Momentum Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['Stochastic_%K'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close']).stoch()
    df['Stochastic_%D'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close']).stoch_signal()

    # 2. Volatility Indicators
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close']).average_true_range()
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()

    # 3. Trend Indicators
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['ADX'] = ta.trend.ADXIndicator(
        df['High'], df['Low'], df['Close']).adx()

    # 4. Volume Indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        df['Close'], df['Volume']).on_balance_volume()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    # 5. Custom Indicators
    df['Price_Volume_Trend'] = calculate_price_volume_trend(df)
    df['Daily_Return'] = df['Close'].pct_change()

    return df


def calculate_price_volume_trend(df: pd.DataFrame) -> pd.Series:
    """Calcule un indicateur personnalisé combinant prix et volume."""
    price_change = df['Close'].pct_change()
    volume_change = df['Volume'].pct_change()
    return (price_change * volume_change).cumsum()