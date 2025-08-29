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

class TestFormatReportToHtml:
    """Test suite pour la méthode format_report_to_html de SignalReporter."""

    def test_format_report_with_buy_and_sell_signals(self):
        """Test le formatage HTML quand il y a des signaux d'achat et de vente."""
        # 1. Arrange - Créer un rapport de test complet
        report = {
            'buy_signals': [
                {'ticker': 'AIR.PA', 'signal': 'BUY', 'price': 150.50, 'date': pd.Timestamp('2023-10-27')},
                {'ticker': 'SAN.PA', 'signal': 'BUY', 'price': 45.75, 'date': pd.Timestamp('2023-10-27')}
            ],
            'sell_signals': [
                {'ticker': 'BNP.PA', 'signal': 'SELL', 'price': 620.80, 'date': pd.Timestamp('2023-10-27')}
            ],
            'hold_signals': [],  # Non utilisé dans le formatage actuel, mais présent
            'errors': [],
            'report_date': dt.date(2023, 10, 27),
            'total_tickers_analyzed': 3
        }

        # 2. Act - Appeler la méthode à tester
        # On crée une instance minimale de SignalReporter pour appeler la méthode
        # On n'a pas besoin de vrais strategy_cls ou data_loader pour ce test
        reporter = SignalReporter(strategy=None, data_loader=None)
        html_output = reporter.format_report_to_html(report)

        # 3. Assert - Vérifier que le HTML contient les éléments attendus
        assert "Rapport de Trading Automatique" in html_output
        assert "2023-10-27" in html_output  # La date du rapport
        assert "3" in html_output  # Le nombre total de tickers analysés

        # Vérifier la présence des signaux d'achat
        assert "Signaux d'ACHAT" in html_output
        assert "AIR.PA" in html_output and "150.50" in html_output
        assert "SAN.PA" in html_output and "45.75" in html_output

        # Vérifier la présence des signaux de vente
        assert "Signaux de VENTE" in html_output
        assert "BNP.PA" in html_output and "620.80" in html_output

        # Vérifier l'absence de messages d'erreur et de "Aucun signal"
        assert "Aucun signal d'achat" not in html_output
        assert "Aucun signal de vente" not in html_output

    def test_format_report_with_no_signals(self):
        """Test le formatage HTML quand il n'y a aucun signal (que des HOLD)."""
        # 1. Arrange - Créer un rapport sans signaux
        report = {
            'buy_signals': [],
            'sell_signals': [],
            'hold_signals': [{'ticker': 'GLE.PA', 'signal': 'HOLD', 'price': 100.0, 'date': pd.Timestamp('2023-10-27')}],
            'errors': [],
            'report_date': dt.date(2023, 10, 27),
            'total_tickers_analyzed': 1
        }

        # 2. Act
        reporter = SignalReporter(strategy=None, data_loader=None)
        html_output = reporter.format_report_to_html(report)

        # 3. Assert - Vérifier les messages "Aucun signal"
        assert "Aucun signal d'achat aujourd'hui" in html_output
        assert "Aucun signal de vente aujourd'hui" in html_output
        # Vérifier que les sections de signaux spécifiques ne sont pas présentes
        assert "Signaux d'ACHAT" not in html_output
        assert "Signaux de VENTE" not in html_output

    def test_format_report_with_errors(self):
        """Test le formatage HTML quand il y a des erreurs de traitement."""
        # 1. Arrange - Créer un rapport avec des erreurs
        report = {
            'buy_signals': [{'ticker': 'AIR.PA', 'signal': 'BUY', 'price': 150.50, 'date': pd.Timestamp('2023-10-27')}],
            'sell_signals': [],
            'hold_signals': [],
            'errors': [
                {'ticker': 'BAD.PA', 'error': 'Timeout lors de la récupération des données'},
                {'ticker': 'FAIL.PA', 'error': 'Division by zero in RSI calculation'}
            ],
            'report_date': dt.date(2023, 10, 27),
            'total_tickers_analyzed': 3 # 1 réussi + 2 en erreur
        }

        # 2. Act
        reporter = SignalReporter(strategy=None, data_loader=None)
        html_output = reporter.format_report_to_html(report)

        # 3. Assert - Vérifier la présence de la section d'erreurs
        assert "Erreurs Rencontrées" in html_output
        assert "BAD.PA" in html_output and "Timeout" in html_output
        assert "FAIL.PA" in html_output and "Division by zero" in html_output
        # Vérifier que le signal réussi est aussi présent
        assert "AIR.PA" in html_output and "150.50" in html_output

    def test_format_report_missing_optional_keys(self):
        """Test la robustesse de la fonction si des clés optionnelles sont manquantes."""
        # 1. Arrange - Créer un rapport minimal sans 'hold_signals' ni 'errors'
        report = {
            'buy_signals': [],
            'sell_signals': [],
            # 'hold_signals' key is missing
            # 'errors' key is missing
            'report_date': dt.date(2023, 10, 27),
            'total_tickers_analyzed': 2
        }

        # 2. Act & Assert - Vérifier que la fonction ne plante pas
        reporter = SignalReporter(strategy=None, data_loader=None)
        # Cela ne devrait pas lever d'exception KeyError
        html_output = reporter.format_report_to_html(report)

        # 3. Vérifier le contenu de base
        assert "Rapport de Trading Automatique" in html_output
        assert "Aucun signal d'achat aujourd'hui" in html_output
        assert "Aucun signal de vente aujourd'hui" in html_output