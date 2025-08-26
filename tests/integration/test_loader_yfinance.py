import pandas as pd
import pytest

from trading_system.data.loader import load_yfinance_data, load_multiple_tickers


def test_integration_load_yfinance_data(ticker_test):
    result = load_yfinance_data('AIR.PA', start_date="2023-01-02", end_date="2023-01-07", interval="1d", progress=False)
    assert not result.empty
    assert len(result) == 5

@pytest.mark.skip(reason="not working yet")
def test_integration_load_multiple_tickers():
    tickers = ["AIR.PA", "SAN.PA", "BNP.PA", "GLE.PA", "ML.PA"]
    result = load_multiple_tickers(tickers, start_date="2023-01-02", end_date="2023-01-07", interval="1d", progress=False)
    for ticker in tickers:
        assert ticker in result
        assert not result[ticker].empty
        assert len(result[ticker]) == 5
