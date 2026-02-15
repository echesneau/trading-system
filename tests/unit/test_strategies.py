import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from trading_system.strategies.classical import ClassicalStrategy
from trading_system.strategies.hybrid import HybridStrategy

def test_hybrid_strategy_buy_signal():
    """Teste un signal d'achat avec la stratégie hybride."""
    # Créer un mock pour le modèle ML
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    # Configurer les valeurs de retour
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 80% de probabilité de hausse
    mock_scaler.transform.return_value = np.array([
        [25,  # RSI bas
         1.5,  # MACD > Signal (positif)
         1.0,  # Signal
         90,  # Prix
         92,  # Bande inférieure (prix < bande)
         101,  # Bande superieur (prix > bande)
         3,  # ATR
         95,  # EMA50
         100,  # EMA200 (EMA50 > EMA200 = tendance haussière)
         1500  # Volume
         ]
    ])

    # Initialiser la stratégie avec les mocks
    model_artifacts = {
        'model': mock_model,
        'scaler': mock_scaler,
        'feature_names': ['RSI', 'MACD', 'MACD_Signal', 'Close', 'BB_Lower',
                          'BB_Upper', 'ATR', 'EMA_50', 'EMA_200', 'VolMA20']
    }
    strategy = HybridStrategy(model_artifacts, rsi_buy=30)

    # Données avec conditions techniques d'achat
    data = pd.DataFrame({
        'RSI': [25, 26, 27, 28, 29],
        'MACD': [1.5, 1.4, 1.3, 1.2, 1.1],
        'MACD_Signal': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Close': [95, 94, 93, 92, 91],
        'BB_Upper': [96, 95, 94, 93, 92],
        'BB_Lower': [96, 95, 94, 93, 92],
        'ATR': [3, 3, 3, 3, 3],
        'EMA_50': [100, 99, 98, 97, 96],
        'EMA_200': [105, 104, 103, 102, 101],
        'VolMA20': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Générer les signaux
    signals = strategy.generate_signals(data)
    
    # Vérifier le signal d'achat
    assert signals.iloc[-1] == 'BUY', (
        f"Attendu 'BUY' mais obtenu '{signals.iloc[-1]}'. "
        f"Dernières valeurs:\n{data.iloc[-1]}\n"
        f"Proba modèle: {mock_model.predict_proba.return_value}"
    )
    mock_model.predict_proba.assert_called_once()
    mock_scaler.transform.assert_called_once()


def test_hybrid_strategy_no_buy_with_low_ml_confidence():
    """Vérifie qu'aucun achat n'est déclenché avec une faible confiance ML."""
    # Créer un mock pour le modèle ML
    mock_model = MagicMock()
    mock_scaler = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])  # Seulement 50% de confiance
    mock_scaler.transform.return_value = np.array([
        [25,  # RSI bas
         1.5,  # MACD > Signal (positif)
         1.0,  # Signal
         90,  # Prix
         92,  # Bande inférieure (prix < bande)
         3,  # ATR
         95,  # EMA50
         100,  # EMA200 (EMA50 > EMA200 = tendance haussière)
         1500  # Volume
         ]
    ])
    model_artifacts = {
        'model': mock_model,
        'scaler': mock_scaler,
        'feature_names': ['RSI', 'MACD', 'MACD_Signal', 'Close', 'BB_Lower',
                          'BB_Upper', 'ATR', 'EMA_50', 'EMA_200', 'VolMA20']
    }
    strategy = HybridStrategy(model_artifacts, rsi_buy=30)

    data = pd.DataFrame({
        'RSI': [25, 26, 27, 28, 29],
        'MACD': [1.5, 1.4, 1.3, 1.2, 1.1],
        'MACD_Signal': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Close': [95, 94, 93, 92, 91],
        'BB_Upper': [96, 95, 94, 93, 92],
        'BB_Lower': [96, 95, 94, 93, 92],
        'ATR': [3, 3, 3, 3, 3],
        'EMA_50': [100, 99, 98, 97, 96],
        'EMA_200': [105, 104, 103, 102, 101],
        'VolMA20': [1000, 1100, 1200, 1300, 1400]
    })
    
    signals = strategy.generate_signals(data)
    assert signals.iloc[0] == 'HOLD'

def test_hybrid_strategy_sell_signal():
    """Teste un signal de vente avec la stratégie hybride."""
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    # Configurer les valeurs de retour
    mock_model.predict_proba.return_value = np.array([[0.1, 0.0]])
    mock_scaler.transform.return_value = np.array([
        [80,  # RSI bas
         0.9,  # MACD < Signal (positif)
         1.0,  # Signal
         95,  # Prix
         92,  # Bande inférieure (prix < bande)
         3,  # ATR
         101,  # EMA50
         100,  # EMA200 (EMA50 < EMA200 = tendance baissière)
         1500  # Volume
         ]
    ])
    model_artifacts = {
        'model': mock_model,
        'scaler': mock_scaler,
        'feature_names': ['RSI', 'MACD', 'MACD_Signal', 'Close', 'BB_Lower',
                          'BB_Upper', 'ATR', 'EMA_50', 'EMA_200', 'VolMA20']
    }
    strategy = HybridStrategy(model_artifacts, rsi_buy=30, rsi_sell=70)
    
    # Données avec conditions techniques de vente
    data = pd.DataFrame({
        'RSI': [71, 75, 75, 79, 80],
        'MACD': [1.5, 1.4, 1.3, 1.2, 1.1],
        'MACD_Signal': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Close': [95, 94, 93, 92, 91],
        'BB_Upper': [96, 95, 94, 93, 92],
        'BB_Lower': [96, 95, 94, 93, 92],
        'ATR': [3, 3, 3, 3, 3],
        'EMA_50': [100, 99, 98, 97, 96],
        'EMA_200': [105, 104, 103, 102, 101],
        'VolMA20': [1000, 1100, 1200, 1300, 1400]
    })
    signals = strategy.generate_signals(data)
    assert signals.iloc[0] == 'SELL'

def test_cond_disabled_and_does_not_block_buy():
    data = pd.DataFrame({
        "RSI": [20, 50, 80],
        "MACD": [1.0, -1.0, 0.5],
        "MACD_Signal": [0.0, 0.0, 0.0],
        "Close": [90, 100, 110],
        "BB_Lower": [95, 95, 95],
        "BB_Upper": [105, 105, 105],
        "ADX": [25, 25, 25],
        "Stochastic_%K": [np.nan, np.nan, np.nan],
        "Stochastic_%D": [np.nan, np.nan, np.nan],
        "ATR": [3, 3, 3],
    })
    strat = ClassicalStrategy(rsi_buy=None, rsi_sell=70)
    signals = strat.generate_signals(data)
    # RSI=None ne doit PAS empêcher un BUY
    # index 0 : MACD > signal & Close < BB_Lower
    assert signals.iloc[0] == 1

def test_cond_disabled_or_does_not_trigger_sell_everywhere():
    data = pd.DataFrame({
        "RSI": [20, 50, 80],
        "MACD": [1.0, -1.0, 0.5],
        "MACD_Signal": [0.0, 0.0, 0.0],
        "Close": [90, 100, 110],
        "BB_Lower": [95, 95, 95],
        "BB_Upper": [105, 105, 105],
        "ADX": [25, 25, 25],
        "Stochastic_%K": [np.nan, np.nan, np.nan],
        "Stochastic_%D": [np.nan, np.nan, np.nan],
        "ATR": [3, 3, 3],
    })

    strat = ClassicalStrategy(
        rsi_buy=30,
        rsi_sell=None     # désactivé dans un OR
    )

    signals = strat.generate_signals(data)

    # Avant le fix : tout était SELL (-1)
    # Après le fix : SELL seulement quand l'autre branche est vraie
    assert signals.iloc[0] != -1
    assert signals.iloc[1] != -1

def test_sell_triggered_by_macd_and_bb():
    data = pd.DataFrame({
        "RSI": [20, 50, 80],
        "MACD": [1.0, -1.0, 0.5],
        "MACD_Signal": [0.0, 0.0, 0.0],
        "Close": [90, 106, 110],
        "BB_Lower": [95, 95, 95],
        "BB_Upper": [105, 105, 105],
        "ADX": [25, 25, 25],
        "Stochastic_%K": [np.nan, np.nan, np.nan],
        "Stochastic_%D": [np.nan, np.nan, np.nan],
        "ATR": [3, 3, 3],
    })

    strat = ClassicalStrategy(
        rsi_buy=30,
        rsi_sell=None
    )
    signals = strat.generate_signals(data)

    # index 1 :
    # MACD < signal AND Close > BB_Upper → SELL
    assert signals.iloc[1] == -1

def test_signal_values_are_valid():
    data = pd.DataFrame({
        "RSI": [20, 50, 80],
        "MACD": [1.0, -1.0, 0.5],
        "MACD_Signal": [0.0, 0.0, 0.0],
        "Close": [90, 100, 110],
        "BB_Lower": [95, 95, 95],
        "BB_Upper": [105, 105, 105],
        "ADX": [25, 25, 25],
        "Stochastic_%K": [np.nan, np.nan, np.nan],
        "Stochastic_%D": [np.nan, np.nan, np.nan],
        "ATR": [3, 3, 3],
    })
    strat = ClassicalStrategy(
        rsi_buy=30,
        rsi_sell=70
    )

    signals = strat.generate_signals(data)

    assert set(signals.unique()).issubset({-1, 0, 1})
    assert len(signals) == len(data)