# tests/unit/test_data_loader.py
import pytest
from unittest.mock import patch
import pandas as pd

from trading_system.data import load_yfinance_data, DataLoadingError

def test_load_yfinance_data_success():
    """Test le chargement réussi de données"""
    with patch('yfinance.download') as mock_download:
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1200]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))

        mock_download.return_value = mock_data

        result = load_yfinance_data("AIR.PA", start_date="2023-01-01")

        assert not result.empty
        assert set(['Open', 'High', 'Close', 'Low', 'Volume']).issubset(result.columns)


def test_load_yfinance_data_empty_data():
    """Test la gestion des données vides"""
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(DataLoadingError) as excinfo:
            load_yfinance_data("INVALID.TICKER", start_date="2023-01-01")

        assert "Aucune donnée" in str(excinfo.value)


def test_load_yfinance_data_api_failure():
    """Test que toute exception API est convertie en DataLoadingError"""

    # Cas 1: Exception générique
    with patch('yfinance.download', side_effect=Exception("Mocked API failure")):
        with pytest.raises(DataLoadingError) as excinfo:
            load_yfinance_data("AIR.PA", "2023-01-01")
        assert "Erreur avec Yahoo Finance" in str(excinfo.value)

    # Cas 2: Erreur réseau spécifique
    with patch('yfinance.download', side_effect=ConnectionError("Timeout")):
        with pytest.raises(DataLoadingError) as excinfo:
            load_yfinance_data("AIR.PA", "2023-01-01")
        assert "Timeout" in str(excinfo.value)

    # Cas 3: Erreur de valeur
    with patch('yfinance.download', side_effect=ValueError("Invalid parameter")):
        with pytest.raises(DataLoadingError) as excinfo:
            load_yfinance_data("AIR.PA", "2023-01-01")
        assert "Invalid parameter" in str(excinfo.value)


def test_load_yfinance_data_invalid_ticker():
    """Test la validation du ticker"""
    with pytest.raises(DataLoadingError):
        load_yfinance_data("", start_date="2023-01-01")  # Ticker vide