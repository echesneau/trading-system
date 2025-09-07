# src/features/technical.py
import pandas as pd
import numpy as np
import ta  # Bibliothèque d'analyse technique
from ta.trend import ema_indicator
from typing import Optional, List


def calculate_indicators(
        data: pd.DataFrame,
        rsi_window: int = 14,
        atr_window: int = 14,
        adx_window: int = 14,
        ema_windows: List[int] = [5, 10, 20],
        bollinger_window: int = 20,
        bollinger_std: int = 2,
        macd_slow: int = 26,
        macd_fast: int = 12,
        macd_signal: int = 9,
        min_periods: Optional[int] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques pour les données de marché.

    Args:
        data: DataFrame contenant les colonnes OHLCV (Open, High, Low, Close, Volume)
        rsi_window: Période RSI (défaut: 14)
        atr_window: Période ATR (défaut: 14)
        adx_window: Période ADX (défaut: 14)
        ema_windows: Périodes EMA (défaut: [20, 50, 200])
        bollinger_window: Période Bollinger Bands (défaut: 20)
        min_periods: Nombre minimum de périodes requises (None = window size)

    Returns:
        DataFrame original enrichi avec les indicateurs techniques
    """
    df = data.copy()

    # Vérification des colonnes requises
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Données manquantes pour calculer les indicateurs: {missing}")

    # Déterminer la taille de fenêtre maximale
    max_window = max([rsi_window, atr_window, adx_window, bollinger_window] + ema_windows)

    # Vérifier suffisamment de données
    if len(df) < max_window:
        raise ValueError(f"Nécessite au moins {max_window} périodes, {len(df)} fournies")

    # 1. Momentum Indicators
    ## RSI avec gestion des NaN initiaux
    df['RSI'] = ta.momentum.RSIIndicator(
        close=df['Close'],
        window=rsi_window,
        fillna=False
    ).rsi()
    ## MACD (fenêtres fixes conventionnelles)
    macd = ta.trend.MACD(close=df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    ## Stochastic Oscillator
    df['Stochastic_%K'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close']).stoch()
    df['Stochastic_%D'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close']).stoch_signal()

    # 2. Volatility Indicators
    ## ATR
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=atr_window,
        fillna=False
    ).average_true_range()

    # Bandes de Bollinger
    bb = ta.volatility.BollingerBands(
        close=df['Close'],
        window=bollinger_window,
        window_dev=bollinger_std,
        fillna=False
    )
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    bollinger = ta.volatility.BollingerBands(df['Close'])

    # 3. Trend Indicators
    ## Exponential Moving Averages (EMA)
    for window in ema_windows:
        if len(df) >= window:
            df[f'EMA_{window}'] = ema_indicator(df['Close'], window=window)
        else:
            df[f'EMA_{window}'] = np.nan
    ## ADX
    if len(df) >= adx_window:
        df['ADX'] = ta.trend.ADXIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=adx_window,
            fillna=False
        ).adx()
    else:
        df['ADX'] = np.nan

    # 4. Volume Indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        df['Close'], df['Volume']).on_balance_volume()
    df['VolMA20'] = df['Volume'].rolling(window=20).mean()

    # 5. Custom Indicators
    df['Price_Volume_Trend'] = calculate_price_volume_trend(df)
    df['Daily_Return'] = df['Close'].pct_change()

    return df


def calculate_price_volume_trend(df: pd.DataFrame) -> pd.Series:
    """Calcule un indicateur personnalisé combinant prix et volume."""
    price_change = df['Close'].pct_change()
    volume_change = df['Volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    # On utilise la multiplication signée pour garder la direction
    pv_trend = (price_change * volume_change.abs())  # Volume absolu mais garde le signe du prix

    # Cumsum avec reset quand le signe change
    trend = pv_trend.cumsum()

    # Réinitialisation partielle pour éviter la dérive
    trend = trend - trend.rolling(3, min_periods=1).mean()

    return trend