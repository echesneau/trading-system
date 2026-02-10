import pandas as pd
from trading_system.features import calculate_indicators, calculate_price_volume_trend
from trading_system.data.loader import load_yfinance_data, load_ccxt_data

def test_test_custom_indicators():
    data = load_yfinance_data(
        ticker='SAN.PA',
        start_date="2010-01-01",
        end_date="2023-06-30",
        interval="1d"
    )
    result = calculate_price_volume_trend(data)
    assert pd.isnull(result).sum() == 1

def test_calculate_indicators_crypto():
    pair = 'BTC/EUR'
    data = load_ccxt_data("BTC/EUR", exchange_name="kraken", interval="1d",
                            start_date="2023-10-01", end_date="2025-10-02")
    result = calculate_indicators(data, rsi_window=14, atr_window=14,
                                  adx_window=7, ema_windows=[5, 10, 20],
                                  bollinger_window=20, bollinger_std=2,
                                  macd_slow=26, macd_fast=12, macd_signal=9,
                                  volume_ma_window=20, balance_volume=True,
                                  stochastic_oscillator=True, price_volume_trend=True)
    assert not result.empty
    assert len(result) == len(data)
    expected_indicators = [
        'RSI', 'MACD', 'MACD_Signal', 'Stochastic_%K', 'Stochastic_%D',
        'ATR', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'EMA_5', 'EMA_10', 'EMA_20', 'ADX',
        'OBV', 'VolMA', 'Price_Volume_Trend', 'Daily_Return'
    ]

    for indicator in expected_indicators:
        assert indicator in result.columns, f"{indicator} manquant dans les résultats"

    # Vérifier que les valeurs sont calculées
    assert not result['RSI'].isna().all()
    assert not result['MACD'].isna().all()
    assert not result['ATR'].isna().all()
    assert result['BB_Upper'].iloc[-1] > result['BB_Lower'].iloc[-1]
    assert result['MACD'].iloc[-1] != 0
