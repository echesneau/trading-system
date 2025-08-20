import pytest
from src.trading_system.backtesting import BacktestingEngine
from src.trading_system.strategies.classical import ClassicalStrategy
from src.trading_system.strategies.hybrid import HybridStrategy
from src.trading_system.features import calculate_indicators


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

