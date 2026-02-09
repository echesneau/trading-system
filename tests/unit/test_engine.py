# tests/unit/test_engine.py
import numpy as np
import pandas as pd
import pytest
from trading_system.backtesting import BacktestingEngine
from trading_system.backtesting.engine import backtest_core


def test_backtest_core_simple_trade():
    # --- Données de test ---
    prices = np.array([100., 100., 100., 100.])
    signals = np.array([1, 0, -1, 0])  # BUY puis SELL

    initial_capital = 1000.0
    position_size = 1.0
    fee = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    # --- Exécution ---
    portfolio_values, positions, trades = backtest_core(
        prices,
        signals,
        initial_capital,
        position_size,
        fee,
        stop_loss,
        take_profit
    )

    # --- Assertions structurelles ---
    assert portfolio_values.shape == (4,)
    assert positions.shape == (4,)
    assert trades.shape == (4, 4)

    # --- Vérification du BUY ---
    assert trades[0, 0] == 1          # action BUY
    assert trades[0, 1] == 100.0      # prix
    assert trades[0, 2] == 10.0       # 1000 / 100
    assert trades[0, 3] == 0          # reason = signal

    # --- Vérification de la position ---
    assert positions[0] == 10
    assert positions[1] == 10
    assert positions[2] == 0          # après SELL

    # --- Vérification du SELL ---
    assert trades[2, 0] == -1         # action SELL
    assert trades[2, 1] == 100.0
    assert trades[2, 2] == 10.0
    assert trades[2, 3] == 0          # reason = signal

    # --- Capital final ---
    assert portfolio_values[-1] == pytest.approx(1000.0)

def test_backtest_core_stop_loss():
    prices = np.array([100., 95., 90.])
    signals = np.array([1, 0, 0])

    portfolio_values, positions, trades = backtest_core(
        prices,
        signals,
        initial_capital=1000.0,
        position_size=1.0,
        fee=0.0,
        stop_loss=0.96,   # déclenché à 95
        take_profit=0.0
    )

    # Stop-loss déclenché à l'index 1
    assert trades[1, 0] == -1
    assert trades[1, 3] == 1          # reason = stop_loss
    assert positions[2] == 0


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
        {"action": 1, "price": 100},
        {"action": -1, "price": 120},
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
        {"action": 1, "price": 100},
        {"action": -1, "price": 80},
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
