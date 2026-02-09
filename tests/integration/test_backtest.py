import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import MagicMock

from trading_system.backtesting import BacktestingEngine
from trading_system.data.loader import load_yfinance_data
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.features.technical import calculate_indicators


def test_backtest_engine_basic():
    """Teste le fonctionnement de base du moteur de backtesting."""
    # Créer des données de marché simulées
    data = pd.DataFrame({
        'Close': [100, 102, 105, 103, 107, 110, 108, 112],
    }, index=pd.date_range('2023-01-01', periods=8))
    
    # Créer une stratégie mock
    strategy = MagicMock()
    strategy.generate_signals.return_value = pd.Series(
        ['HOLD', 'BUY', 'HOLD', 'SELL', 'HOLD', 'BUY', 'HOLD', 'SELL'],
        index=data.index
    )
    
    # Configurer le moteur de backtest
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=10000,
        transaction_fee=0.001
    )
    
    # Exécuter le backtest
    results = engine.run()
    
    # Vérifier les résultats
    assert isinstance(results, dict)
    assert 'portfolio' in results
    assert 'performance' in results
    assert 'trades' in results
    
    # Vérifier la valeur finale du portefeuille
    assert results['portfolio']['value'].iloc[-1] > 10000
    
    # Vérifier les transactions
    trades = results['trades']
    assert len(trades) == 4  # 2 trades complets (achat + vente)
    assert all(col in trades.columns for col in ["action", "price", "shares"])
    
    # Premier trade: achat à 102, vente à 103
    assert trades.iloc[0]['action'] == 'BUY'
    assert trades.iloc[0]['price'] == 102
    assert trades.iloc[1]['action'] == 'SELL'
    assert trades.iloc[1]['price'] == 103

def test_position_sizing():
    """Teste la gestion de la taille des positions."""
    data = pd.DataFrame({
        'Close': [100, 101, 102, 103],
    }, index=pd.date_range('2023-01-01', periods=4))
    
    strategy = MagicMock()
    strategy.generate_signals.return_value = pd.Series(
        ['BUY', 'HOLD', 'HOLD', 'SELL']
    )
    
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=10000,
        position_size=0.5  # 50% du capital par position
    )
    
    results = engine.run()
    trades = results['trades']
    
    # Vérifier que la taille de la position est correcte
    shares_bought = trades.iloc[0]['shares']
    assert abs(shares_bought * 100 - 5000) < 1  # 50% de 10000 à 100€

    # Vérifie qu'il y a bien un SELL correspondant
    assert "SELL" in trades["action"].values

def test_stop_loss_and_take_profit():
    """Teste le déclenchement des stop-loss et take-profit."""
    data = pd.DataFrame({
        'Close': [100, 95, 90, 115, 110]  # Chute puis hausse
    }, index=pd.date_range('2023-01-01', periods=5))
    
    strategy = MagicMock()
    strategy.generate_signals.return_value = pd.Series(
        ['BUY', 'HOLD', 'HOLD', 'HOLD', 'HOLD']
    )
    
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=10000,
        stop_loss=0.9,  # -10%
        take_profit=1.1  # +10%
    )
    
    results = engine.run()
    trades = results['trades']
    
    # Vérifier que le stop-loss a été déclenché
    assert len(trades) >= 2
    assert trades.iloc[1]['action'] == 'SELL'
    assert trades.iloc[1]['price'] == 90  # Prix du stop-loss
    assert trades.iloc[1]['reason'] == 'stop_loss'


def test_no_signals_results_in_no_trades():
    """Vérifie qu'aucun signal ne produit aucun trade."""
    data = pd.DataFrame({
        'Close': [100, 101, 102, 103],
    }, index=pd.date_range('2023-01-01', periods=4))

    strategy = MagicMock()
    strategy.generate_signals.return_value = pd.Series(
        ['HOLD', 'HOLD', 'HOLD', 'HOLD'],
        index=data.index
    )

    engine = BacktestingEngine(strategy=strategy, data=data, initial_capital=10000)
    results = engine.run()

    assert results['trades'].empty
    assert results['portfolio']['value'].iloc[-1] == 10000  # capital inchangé


def test_all_buy_signals():
    """Vérifie le comportement si la stratégie renvoie uniquement des BUY."""
    data = pd.DataFrame({
        'Close': [100, 101, 102, 103],
    }, index=pd.date_range('2023-01-01', periods=4))

    strategy = MagicMock()
    strategy.generate_signals.return_value = pd.Series(
        ['BUY', 'BUY', 'BUY', 'BUY'],
        index=data.index
    )

    engine = BacktestingEngine(strategy=strategy, data=data, initial_capital=10000)
    results = engine.run()

    trades = results['trades']
    # Normalement un seul BUY est exécuté, pas plusieurs d'affilée
    assert trades["action"].tolist().count("BUY") == 1

def test_backtest_classical_strategy_numba(ticker_test):
    raw_data = load_yfinance_data(
        ticker=ticker_test,
        start_date="2022-01-01",
        end_date="2024-01-01",
        interval="1d"
    )
    params = {
        "rsi_window":14,
        "rsi_buy":30,
        "rsi_sell":70,
        "macd_fast":12,
        "macd_slow":26,
        "macd_signal":9,
        "bollinger_window":20,
        'bollinger_std':2,
    }
    data = calculate_indicators(raw_data, **params)

    strategy = ClassicalStrategy(**params)
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=1000,
        transaction_fee=0.05,
        position_size=1
    )
    result = engine.run_numba()
    for k in ['portfolio', 'trades', 'performance']:
        assert k in result

    assert isinstance(result['portfolio'], pd.DataFrame)
    assert isinstance(result['trades'], pd.DataFrame)
    assert isinstance(result['performance'], dict)
    assert len(result['trades']) > 0  # Doit avoir généré des trades

def test_backtest_classical_strategy_without_numba(ticker_test):
    raw_data = load_yfinance_data(
        ticker=ticker_test,
        start_date="2022-01-01",
        end_date="2024-01-01",
        interval="1d"
    )
    params = {
        "rsi_window":14,
        "rsi_buy":30,
        "rsi_sell":70,
        "macd_fast":12,
        "macd_slow":26,
        "macd_signal":9,
        "bollinger_window":20,
        'bollinger_std':2,
    }
    data = calculate_indicators(raw_data, **params)

    strategy = ClassicalStrategy(**params)
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=1000,
        transaction_fee=0.05,
        position_size=1
    )
    result = engine.run()
    for k in ['portfolio', 'trades', 'performance']:
        assert k in result

    assert isinstance(result['portfolio'], pd.DataFrame)
    assert isinstance(result['trades'], pd.DataFrame)
    assert isinstance(result['performance'], dict)
    assert len(result['trades']) > 0  # Doit avoir généré des trades

def test_backtest_classical_strategy_compare_run(ticker_test):
    raw_data = load_yfinance_data(
        ticker=ticker_test,
        start_date="2022-01-01",
        end_date="2024-01-01",
        interval="1d"
    )
    params = {
        "rsi_window":14,
        "rsi_buy":30,
        "rsi_sell":70,
        "macd_fast":12,
        "macd_slow":26,
        "macd_signal":9,
        "bollinger_window":20,
        'bollinger_std':2,
    }
    data = calculate_indicators(raw_data, **params)

    strategy = ClassicalStrategy(**params)
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=1000,
        transaction_fee=0.05,
        position_size=1
    )
    result = engine.run()
    result_numba = engine.run_numba()
    trades = result['trades'].reset_index(drop=True)
    trades_numba = result_numba['trades'].reset_index(drop=True)
    trades_numba = trades_numba[trades.columns]
    assert_frame_equal(
        trades,
        trades_numba,
        check_dtype=False
    )
    portfolio = result['portfolio'].reset_index(drop=True)
    portfolio_numba = result_numba['portfolio'].reset_index(drop=True)
    portfolio_numba = portfolio_numba[portfolio.columns]
    assert_frame_equal(
        portfolio,
        portfolio_numba,
        check_dtype=False,
    )