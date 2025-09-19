# tests/unit/test_engine.py
import pandas as pd
from trading_system.backtesting import BacktestingEngine

def test_calculate_performance_edge_cases():
    engine = BacktestingEngine(strategy=None, data=pd.DataFrame())

    # Cas 1 : Données vides
    result = engine._calculate_performance([], pd.DataFrame())
    assert result["return"] == 0.0
    assert result["annualized_return"] == 0.0
    assert result["max_drawdown"] == 0.0
    assert result["sharpe_ratio"] == 0.0

    # Cas 2 : Une seule valeur
    result = engine._calculate_performance([{"date": pd.Timestamp("2020-01-01"), "value": 100}],
                                           pd.DataFrame())
    assert result["return"] == 0.0
    assert result["annualized_return"] == 0.0
    assert result["max_drawdown"] == 0.0
    assert result["sharpe_ratio"] == 0.0

def test_calculate_performance_gain():
    engine = BacktestingEngine(strategy=None, data=pd.DataFrame())

    values = [
        {"date": pd.Timestamp("2020-01-01"), "value": 100},
        {"date": pd.Timestamp("2020-01-10"), "value": 120},
    ]
    trades = pd.DataFrame([
        {"action": "BUY", "price": 100},
        {"action": "SELL", "price": 120},
    ])

    result = engine._calculate_performance(values, trades)
    assert result["return"] > 0
    assert result["max_drawdown"] == 0
    assert result["trade_metrics"]["n_wins"] == 1
    assert result["trade_metrics"]["n_losses"] == 0
    assert result["trade_metrics"]["win_rate"] == 1.0

def test_calculate_performance_loss():
    engine = BacktestingEngine(strategy=None, data=pd.DataFrame())

    values = [
        {"date": pd.Timestamp("2020-01-01"), "value": 100},
        {"date": pd.Timestamp("2020-01-10"), "value": 80},
    ]
    trades = pd.DataFrame([
        {"action": "BUY", "price": 100},
        {"action": "SELL", "price": 80},
    ])

    result = engine._calculate_performance(values, trades)
    assert result["return"] < 0
    assert result["max_drawdown"] > 0
    assert result["trade_metrics"]["n_wins"] == 0
    assert result["trade_metrics"]["n_losses"] == 1
    assert result["trade_metrics"]["win_rate"] == 0.0

def test_sharpe_zero_when_no_volatility():
    engine = BacktestingEngine(strategy=None, data=pd.DataFrame())

    values = [
        {"date": pd.Timestamp("2020-01-01"), "value": 100},
        {"date": pd.Timestamp("2020-01-02"), "value": 100},
        {"date": pd.Timestamp("2020-01-03"), "value": 100},
    ]
    trades = pd.DataFrame()

    result = engine._calculate_performance(values, trades)
    assert result["sharpe_ratio"] == 0.0
