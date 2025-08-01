import pytest
from src.backtesting.engine import BacktestingEngine
from src.strategies.classical import ClassicalStrategy
from src.strategies.hybrid import HybridStrategy

@pytest.fixture
def test_data():
    # Charger des données de test
    return pd.read_csv('tests/data/test_data.csv', index_col=0, parse_dates=True)

def test_classical_strategy(test_data):
    engine = BacktestingEngine(
        strategy=ClassicalStrategy(),
        data=test_data,
        initial_capital=10000
    )
    results = engine.run()
    assert results['return'] > 0
    assert results['max_drawdown'] < 0.2

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