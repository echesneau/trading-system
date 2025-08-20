# tests/unit/test_engine.py
def test_calculate_performance_edge_cases():
    from trading_system.backtesting import BacktestingEngine

    # Cas 1: Données vides
    assert BacktestingEngine._calculate_performance(None, []) == {
        'return': 0.0, 'annualized_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0
    }

    # Cas 2: Une seule valeur
    assert BacktestingEngine._calculate_performance(None, [{'value': 100}]) == {
        'return': 0.0, 'annualized_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0
    }