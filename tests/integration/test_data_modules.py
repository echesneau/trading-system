# tests/integration/test_data_modules.py
import pytest
from src.data.loader import load_yfinance_data
from src.data.processor import process_market_data


@pytest.mark.integration
def test_full_data_pipeline():
    """Test complet avec normalisation des colonnes"""
    # 1. Chargement avec vérification du MultiIndex
    raw_data = load_yfinance_data(
        ticker="SAN.PA",
        start_date="2023-01-01",
        end_date="2023-03-10"
    )

    # Vérification des colonnes
    assert set(raw_data.columns) == {'Open', 'High', 'Low', 'Close', 'Volume'}

    # 2. Traitement
    processed_data = process_market_data(raw_data)

    # Vérifications finales
    assert not processed_data.empty
    required_columns = {'Close', 'RSI', 'EMA_20', 'BB_Upper'}
    assert required_columns.issubset(processed_data.columns)