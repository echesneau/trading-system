# src/features/technical.py
import pandas as pd
import numpy as np
import ta
from ta.trend import ema_indicator
from typing import List


def calculate_indicators(
        data: pd.DataFrame,
        rsi_window: int = None,
        atr_window: int = None,
        adx_window: int = None,
        ema_windows: List[int] = [],
        bollinger_window: int = None,
        bollinger_std: int = None,
        macd_slow: int = None,
        macd_fast: int = None,
        macd_signal: int = None,
        volume_ma_window: int = None,
        balance_volume: bool = False,
        stochastic_oscillator: bool = False,
        price_volume_trend: bool = False,
        cache: dict = {},
        **kwargs
) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques pour les données de marché.

    Args:
        data: DataFrame contenant les colonnes OHLCV (Open, High, Low, Close, Volume)
        rsi_window: Période RSI (défaut: None = Not calculated)
        atr_window: Période ATR (défaut: None = Not calculated)
        adx_window: Période ADX (défaut: None = Not calculated)
        ema_windows: Périodes EMA (défaut: [])
        bollinger_window: Période Bollinger Bands (défaut: None = Not calculated)
        volume_ma_window: Période pour la moyenne mobile du volume (défaut: None = Not calculated)

    Returns:
        DataFrame original enrichi avec les indicateurs techniques
    """
    out = {}
    # Vérification des colonnes requises
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Données manquantes pour calculer les indicateurs: {missing}")

    # Déterminer la taille de fenêtre maximale
    windows = [val for val in [rsi_window, atr_window, adx_window, bollinger_window] if val is not None]
    windows += [val for val in ema_windows if val is not None]
    max_window = max(windows)

    # Vérifier suffisamment de données
    if len(data) < max_window:
        raise ValueError(f"Nécessite au moins {max_window} périodes, {len(data)} fournies")

    # 1. Momentum Indicators
    ## RSI avec gestion des NaN initiaux
    if rsi_window is None:
        out['RSI'] = np.nan
    else:
        if f"RSI_{rsi_window}" in cache:
            out['RSI'] = cache[f"RSI_{rsi_window}"]
        else:
            out['RSI'] = ta.momentum.RSIIndicator(
                close=data['Close'],
                window=rsi_window,
                fillna=False
            ).rsi()
    ## MACD (fenêtres fixes conventionnelles)
    if macd_fast is None or macd_slow is None or macd_signal is None:
        out['MACD'] = np.nan
        out['MACD_Signal'] = np.nan
    else:
        if f"MACD_{macd_fast}_{macd_slow}_{macd_signal}" in cache:
            out['MACD'] = cache[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
            out['MACD_Signal'] = cache[f"MACD_Signal_{macd_fast}_{macd_slow}_{macd_signal}"]
        else:
            macd = ta.trend.MACD(close=data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
            out['MACD'] = macd.macd()
            out['MACD_Signal'] = macd.macd_signal()

    ## Stochastic Oscillator
    if stochastic_oscillator:
        out['Stochastic_%K'] = ta.momentum.StochasticOscillator(
            data['High'], data['Low'], data['Close']).stoch()
        out['Stochastic_%D'] = ta.momentum.StochasticOscillator(
            data['High'], data['Low'], data['Close']).stoch_signal()
    else:
        out['Stochastic_%K'] = np.nan
        out['Stochastic_%D'] = np.nan

    # 2. Volatility Indicators
    ## ATR
    if atr_window is None:
        out['ATR'] = np.nan
    else:
        if f"ATR_{atr_window}" in cache:
            out['ATR'] = cache[f"ATR_{atr_window}"]
        else:
            out['ATR'] = ta.volatility.AverageTrueRange(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=atr_window,
                fillna=False
            ).average_true_range()

    # Bandes de Bollinger
    if bollinger_window is None or bollinger_std is None:
        out['BB_Upper'] = np.nan
        out['BB_Middle'] = np.nan
        out['BB_Lower'] = np.nan
    else:
        if f"BB_Upper_{bollinger_window}_{bollinger_std}" in cache:
            out['BB_Upper'] = cache[f"BB_Upper_{bollinger_window}_{bollinger_std}"]
            out['BB_Middle'] = cache[f"BB_Middle_{bollinger_window}_{bollinger_std}"]
            out['BB_Lower'] = cache[f"BB_Lower_{bollinger_window}_{bollinger_std}"]
        else:
            bb = ta.volatility.BollingerBands(
                close=data['Close'],
                window=bollinger_window,
                window_dev=bollinger_std,
                fillna=False
            )
            out['BB_Upper'] = bb.bollinger_hband()
            out['BB_Middle'] = bb.bollinger_mavg()
            out['BB_Lower'] = bb.bollinger_lband()

    # 3. Trend Indicators
    ## Exponential Moving Averages (EMA)
    for window in ema_windows:
        if len(data) >= window:
            if f"EMA_{window}" in cache:
                out[f'EMA_{window}'] = cache[f"EMA_{window}"]
            else:
                out[f'EMA_{window}'] = ema_indicator(data['Close'], window=window)
        else:
            out[f'EMA_{window}'] = np.nan
    ## ADX
    if not adx_window is None and len(data) >= adx_window:
        if f"ADX_{adx_window}" in cache:
            out['ADX'] = cache[f"ADX_{adx_window}"]
        else:
            out['ADX'] = ta.trend.ADXIndicator(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=adx_window,
                fillna=False
            ).adx()
    else:
        out['ADX'] = np.nan

    # 4. Volume Indicators
    if balance_volume:
        if f"OBV" in cache:
            out['OBV'] = cache["OBV"]
        else:
            out['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                data['Close'], data['Volume']).on_balance_volume()
    else:
        out['OBV'] = np.nan
    if volume_ma_window is not None and len(data) >= volume_ma_window:
        if f"VolMA_{volume_ma_window}" in cache:
            out['VolMA'] = cache[f"VolMA_{volume_ma_window}"]
        else:
            out['VolMA'] = data['Volume'].rolling(window=volume_ma_window).mean()
    else:
        out['VolMA'] = np.nan

    # 5. Custom Indicators
    if price_volume_trend:
        if f"Price_Volume_Trend" in cache:
            out['Price_Volume_Trend'] = cache["Price_Volume_Trend"]
        else:
            out['Price_Volume_Trend'] = calculate_price_volume_trend(data)
    else:
        out['Price_Volume_Trend'] = np.nan
    out['Daily_Return'] = data['Close'].pct_change()

    data = data.assign(**out)
    return data


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