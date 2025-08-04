import pytest
import pandas as pd
from src.backtesting.engine import BacktestingEngine
from src.strategies.classical import ClassicalStrategy
from unittest.mock import MagicMock

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
    assert 'portfolio' in results
    assert 'performance' in results
    assert 'trades' in results
    
    # Vérifier la valeur finale du portefeuille
    assert results['portfolio']['value'].iloc[-1] > 10000
    
    # Vérifier les transactions
    trades = results['trades']
    assert len(trades) == 4  # 2 trades complets (achat + vente)
    
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
    assert trades.iloc[1]['action'] == 'SELL'
    assert trades.iloc[1]['price'] == 90  # Prix du stop-loss
    assert trades.iloc[1]['reason'] == 'stop_loss'
