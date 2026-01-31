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
    result = load_ccxt_data("BTC/EUR", exchange_name="kraken", interval="1d",
                            start_date="2024-10-02", end_date="2024-10-07")
    assert not result.empty
    assert len(result) == 6
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        assert col in result.columns

@pytest.mark.skip(reason="not working yet")
@pytest.mark.parametrize("interval,expected_delta", [
    ("1d", pd.Timedelta(days=1)),
    ("6h", pd.Timedelta(hours=6)),
    ("1h", pd.Timedelta(hours=1)),
    ("15m", pd.Timedelta(minutes=15)),
])
def test_load_ccxt_data_interval(interval, expected_delta):
    df = load_ccxt_data(
        exchange_name="kraken",
        pair="BTC/EUR",
        start_date="2023-10-01",
        end_date="2023-10-10",
        interval=interval,
    )

    # Vérifier qu'on a bien un DataFrame non vide
    assert not df.empty, "Le DataFrame retourné est vide"

    # Vérifier que l'index est bien de type datetime
    assert isinstance(df.index, pd.DatetimeIndex), "L'index doit être un DatetimeIndex"

    # Calcul des deltas entre deux points
    deltas = df.index.to_series().diff().dropna().unique()

    # Vérifie que toutes les différences correspondent à l'interval attendu
    assert all(d == expected_delta for d in deltas), f"Deltas {deltas} ne correspondent pas à {expected_delta}"