# tests/unit/test_notifications.py
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
import datetime as dt
from trading_system.notifications.reporter import SignalReporter
import logging

# Fixture pour un faux data_loader (FONCTION)
@pytest.fixture
def mock_data_loader():
    """Mock une fonction data_loader."""
    def _loader(ticker, start_date, end_date):
        # Retourne un DataFrame de test basique
        index = pd.date_range(start=start_date, periods=20, freq='D') # 20 jours de data
        data = pd.DataFrame({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 102, 'Volume': 1000
        }, index=index)
        return data
    return _loader

# Fixture pour une fausse stratégie Hybride
@pytest.fixture
def mock_hybrid_strategy_cls():
    mock_cls = Mock()
    mock_instance = Mock()
    mock_cls.return_value = mock_instance
    mock_cls.__name__ = 'HybridStrategy'  # <-- Critical for the check in generate_daily_report
    mock_instance.generate_signals.return_value = pd.Series(['HOLD'] * 20) # Signal pour chaque jour
    return mock_cls

# Fixture pour une fausse stratégie Classique
@pytest.fixture
def mock_classical_strategy_cls():
    mock_cls = Mock()
    mock_instance = Mock()
    mock_cls.return_value = mock_instance
    mock_cls.__name__ = 'ClassicalStrategy'  # <-- Different name
    mock_instance.generate_signals.return_value = pd.Series(['HOLD'] * 20)
    return mock_cls

# Fixture pour le reporter avec stratégie hybride
@pytest.fixture
def hybrid_reporter(mock_hybrid_strategy_cls, mock_data_loader):
    return SignalReporter(mock_hybrid_strategy_cls, mock_data_loader)

# Fixture pour le reporter avec stratégie classique
@pytest.fixture
def classical_reporter(mock_classical_strategy_cls, mock_data_loader):
    return SignalReporter(mock_classical_strategy_cls, mock_data_loader)

def test_generate_daily_report_hybrid(hybrid_reporter, mock_hybrid_strategy_cls, mock_data_loader):
    """Test le flux Hybride: doit appeler load_model."""
    tickers = ['TEST.PA']
    model_path = '/fake/path/model.joblib'

    # Mock load_model pour éviter une erreure réelle
    with patch('trading_system.notifications.reporter.load_model') as mock_load_model:
        mock_model_artifacts = {'model': Mock(), 'scaler': Mock(), 'feature_names': ['RSI', 'MACD']}
        mock_load_model.return_value = mock_model_artifacts

        # Execution
        report = hybrid_reporter.generate_daily_report(tickers, model_path=model_path, rsi_buy=30, adx_window=2)

    # Vérifications
    assert report['total_tickers_analyzed'] == 1
    mock_load_model.assert_called_once_with(model_path)
    # Vérifie que la stratégie est initialisée avec les bons paramètres
    mock_hybrid_strategy_cls.assert_called_once_with(mock_model_artifacts, rsi_buy=30, adx_window=2)
    # Vérifie que generate_signals est appelé
    mock_hybrid_strategy_cls.return_value.generate_signals.assert_called_once()

def test_generate_daily_report_classical(classical_reporter, mock_classical_strategy_cls):
    """Test le flux Classique: ne doit PAS appeler load_model."""
    tickers = ['TEST.PA']

    # Mock load_model pour s'assurer qu'il n'est PAS appelé
    with patch('trading_system.notifications.reporter.load_model') as mock_load_model:
        # Execution
        report = classical_reporter.generate_daily_report(tickers, rsi_buy=30, adx_window=2)

    # Vérifications
    assert report['total_tickers_analyzed'] == 1
    mock_load_model.assert_not_called()  # <-- Critical assertion
    mock_classical_strategy_cls.assert_called_once_with(rsi_buy=30, adx_window=2)

def test_generate_daily_report_with_data_loader_error(hybrid_reporter, caplog):
    """Test la gestion d'erreur du data_loader."""
    tickers = ['ERROR.PA']

    # Créer un data_loader qui échoue
    def failing_data_loader(ticker, start_date, end_date):
        raise ConnectionError("API unreachable")

    hybrid_reporter.data_loader = failing_data_loader

    # Exécution
    with caplog.at_level(logging.ERROR):
        report = hybrid_reporter.generate_daily_report(tickers, model_path='/fake/path')

    # Vérifications
    assert report['total_tickers_analyzed'] == 1
    assert len(report['errors']) == 1
    assert report['errors'][0]['ticker'] == 'ERROR.PA'
    assert "API unreachable" in report['errors'][0]['error']
    # Vérifie que l'erreur est bien loggée
    assert "Erreur lors du traitement de ERROR.PA" in caplog.text

# ... (Add tests for buy/sell signals, empty data, etc.)