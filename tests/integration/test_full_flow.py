import pytest
from trading_system.backtesting import BacktestingEngine
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.strategies.hybrid import HybridStrategy
from trading_system.features import calculate_indicators
from trading_system.data.loader import load_ccxt_data, load_yfinance_data
from trading_system.notifications.reporter import SignalReporter


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

def test_classical_strategy_crypto():
    data = load_ccxt_data("BTC/USDT", exchange_name="kraken", interval="1d",
                            start_date="2023-10-01", end_date="2025-08-31")
    processed_data = calculate_indicators(data, ema_windows=[5, 10, 20])
    engine = BacktestingEngine(
        strategy=ClassicalStrategy(rsi_buy=30, rsi_sell=70),
        data=processed_data,
        initial_capital=10000
    )
    results = engine.run()
    assert isinstance(results['performance']['return'], float)
    assert -0.5 < results['performance']['return'] < 2.0
    assert 0 <= results['performance']['max_drawdown'] < 0.5

def test_classical_strategy_with_buy_expected(test_data):
    """
    Thanks to backtesting,
    A buy is expected 2025-12-08 for ERF.PA
    """
    config = {"ERF.PA": {
            "rsi_window": 7,
            "rsi_buy": 35,
            "rsi_sell": 70,
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 13,
            "bollinger_window": 25,
            "bollinger_std": 1
        }
    }
    date = "2025-12-08"
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_yfinance_data,
                              debug=True, debug_date=date)
    report = reporter.generate_daily_report(list(config.keys()), config, max_window_range=100)
    buy_signals = report["buy_signals"]
    hold_signals = report["hold_signals"]
    html_report = reporter.format_report_to_html(report)
    assert len(hold_signals) == 0
    assert len(buy_signals) == 1
    assert "Aucun signal de conservation aujourd'hui." in html_report
    assert "ERF.PA" in html_report

    config = {"CAP.PA": {
        "rsi_window": 7,
        "rsi_buy": 35,
        "rsi_sell": 70,
        "macd_fast": 20,
        "macd_slow": 26,
        "macd_signal": 13,
        "bollinger_window": 20,
        "bollinger_std": 1
        }
    }
    date = "2025-09-01"
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_yfinance_data,
                              debug=True, debug_date=date)
    report = reporter.generate_daily_report(list(config.keys()), config, max_window_range=100)
    buy_signals = report["buy_signals"]
    hold_signals = report["hold_signals"]
    html_report = reporter.format_report_to_html(report)
    assert len(hold_signals) == 0
    assert len(buy_signals) == 1
    assert "Aucun signal de conservation aujourd'hui." in html_report
    assert "CAP.PA" in html_report

def test_classical_strategy_with_sell_expected(test_data):
    """
    Thanks to backtesting,
    A buy is expected 2025-12-18 for ERF.PA
    """
    config = {"ERF.PA": {
            "rsi_window": 7,
            "rsi_buy": 35,
            "rsi_sell": 70,
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 13,
            "bollinger_window": 25,
            "bollinger_std": 1
        }
    }
    date = "2025-12-18"
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_yfinance_data,
                              debug=True, debug_date=date)
    report = reporter.generate_daily_report(list(config.keys()), config, max_window_range=100)
    buy_signals = report["buy_signals"]
    hold_signals = report["hold_signals"]
    sell_signals = report["sell_signals"]
    html_report = reporter.format_report_to_html(report)
    assert len(hold_signals) == 0
    assert len(buy_signals) == 0
    assert len(sell_signals) == 1
    assert "Aucun signal de conservation aujourd'hui." in html_report
    assert "ERF.PA" in html_report

    config = {"CAP.PA": {
        "rsi_window": 7,
        "rsi_buy": 35,
        "rsi_sell": 70,
        "macd_fast": 20,
        "macd_slow": 26,
        "macd_signal": 13,
        "bollinger_window": 20,
        "bollinger_std": 1
        }
    }
    date = "2025-10-21"
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_yfinance_data,
                              debug=True, debug_date=date)
    report = reporter.generate_daily_report(list(config.keys()), config, max_window_range=100)
    buy_signals = report["buy_signals"]
    hold_signals = report["hold_signals"]
    sell_signals = report["sell_signals"]
    html_report = reporter.format_report_to_html(report)
    assert len(hold_signals) == 0
    assert len(buy_signals) == 0
    assert len(sell_signals) == 1
    assert "Aucun signal de conservation aujourd'hui." in html_report
    assert "CAP.PA" in html_report