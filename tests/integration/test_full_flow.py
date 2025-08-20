import pytest
from src.backtesting.engine import BacktestingEngine
from src.strategies.classical import ClassicalStrategy
from src.strategies.hybrid import HybridStrategy
from src.features.technical import calculate_indicators


def test_classical_strategy(test_data):
    processed_data = calculate_indicators(test_data, ema_windows=[5, 10, 20])
    engine = BacktestingEngine(
        strategy=ClassicalStrategy(rsi_buy=30, rsi_sell=70),
        data=processed_data,
        initial_capital=10000
    )
    results = engine.run()
    assert isinstance(results['performance']['return'], float)
    assert -0.5 < results['performance']['return'] < 2.0
    assert 0 <= results['performance']['max_drawdown'] < 0.5

def test_hybrid_strategy(test_data, trained_model_artifacts):
    processed_data = calculate_indicators(
        test_data,
        **trained_model_artifacts.get('technical_params', {})
    )
    # Vérifier la cohérence des features
    required_features = set(trained_model_artifacts['feature_names'])
    available_features = set(processed_data.columns)
    missing_features = required_features - available_features

    if missing_features:
        pytest.skip(f"Features manquantes: {missing_features}. "
                    f"Disponibles: {list(available_features)[:10]}...")
    engine = BacktestingEngine(
        strategy=HybridStrategy(
            trained_model_artifacts,
            rsi_buy=30,
            rsi_sell=70
        ),
        data=processed_data,
        initial_capital=10000
    )
    results = engine.run()

    assert isinstance(results['performance']['return'], float)
    assert -0.5 < results['performance']['return'] < 2.0
    assert 0 <= results['performance']['max_drawdown'] < 0.5

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