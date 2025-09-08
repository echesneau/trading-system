import pandas as pd
import pytest

from trading_system.data.loader import (load_yfinance_data, load_multiple_tickers,
                                        load_kraken_data, load_ccxt_data)


def test_integration_load_yfinance_data(ticker_test):
    result = load_yfinance_data(ticker_test, start_date="2023-01-02", end_date="2023-01-07", interval="1d", progress=False)
    assert not result.empty
    assert len(result) == 5
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        assert col in result.columns

@pytest.mark.skip(reason="not working yet")
def test_integration_load_multiple_tickers():
    tickers = ["AIR.PA", "SAN.PA", "BNP.PA", "GLE.PA", "ML.PA"]
    result = load_multiple_tickers(tickers, start_date="2023-01-02", end_date="2023-01-07", interval="1d", progress=False)
    for ticker in tickers:
        assert ticker in result
        assert not result[ticker].empty
        assert len(result[ticker]) == 5

def test_integration_load_kraken_data():
    result = load_kraken_data('XXBTZEUR', start_date="2025-01-02", end_date="2025-01-07", interval="1d")
    assert not result.empty
    assert len(result) == 5
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        assert col in result.columns

def test_integration_load_ccxt_data():
    result = load_ccxt_data("BTC/USDT", exchange_name="binance", interval="1d",
                            start_date="2023-01-02", end_date="2023-01-07")
    assert not result.empty
    assert len(result) == 5
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        assert col in result.columns