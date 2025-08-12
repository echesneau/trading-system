import pytest
import pandas as pd
from src.data.loader import load_yfinance_data
from src.backtesting.engine import BacktestingEngine
from src.strategies.classical import ClassicalStrategy
from src.strategies.hybrid import HybridStrategy
from src.features.technical import calculate_indicators


@pytest.fixture
def ticker_test():
    """Fixture pour le ticker de test"""
    return "SAN.PA"

@pytest.fixture
def test_data(ticker_test):
    # Charger des données de test
    return load_yfinance_data(
        ticker=ticker_test,
        start_date="2023-01-01",
        end_date="2023-06-01"
    )

def test_classical_strategy(test_data):
    processed_data = calculate_indicators(test_data, ema_windows=[5, 10, 20])
    engine = BacktestingEngine(
        strategy=ClassicalStrategy(rsi_buy=30, rsi_sell=70),
        data=processed_data,
        initial_capital=10000
    )
    results = engine.run()
    assert isinstance(results['performance']['return'], float)
    assert -0.5 < results['performance']['return'] < 2.0  # Plage large
    assert 0 <= results['performance']['max_drawdown'] < 0.5

def test_hybrid_strategy(test_data):
    engine = BacktestingEngine(
        strategy=HybridStrategy(config={'model_path': 'models/rf_model.joblib'}),
        data=test_data,
        initial_capital=10000
    )
    results = engine.run()
    assert results['return'] > 0
    assert results['max_drawdown'] < 0.15

def test_strategy_comparison(test_data):
    classical_engine = BacktestingEngine(
        strategy=ClassicalStrategy(),
        data=test_data
    )
    hybrid_engine = BacktestingEngine(
        strategy=HybridStrategy(config={'model_path': 'models/rf_model.joblib'}),
        data=test_data
    )
    
    classical_results = classical_engine.run()
    hybrid_results = hybrid_engine.run()
    
    # Vérifier que la stratégie hybride surperforme
    assert hybrid_results['return'] > classical_results['return']
    assert hybrid_results['sharpe'] > classical_results['sharpe']