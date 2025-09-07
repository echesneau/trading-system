import pandas as pd
from trading_system.features import calculate_indicators, calculate_price_volume_trend
from trading_system.data.loader import load_yfinance_data

def test_test_custom_indicators():
    data = load_yfinance_data(
        ticker='SAN.PA',
        start_date="2010-01-01",
        end_date="2023-06-30",
        interval="1d"
    )
    result = calculate_price_volume_trend(data)
    assert pd.isnull(result).sum() == 1
