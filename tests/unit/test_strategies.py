import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.strategies.hybrid import HybridStrategy

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
         3,  # ATR
         95,  # EMA50
         100,  # EMA200 (EMA50 > EMA200 = tendance haussière)
         1500  # Volume
         ]
    ])

    # Initialiser la stratégie avec les mocks
    strategy = HybridStrategy(model=mock_model, scaler=mock_scaler, rsi_buy=30)

    # Données avec conditions techniques d'achat
    data = pd.DataFrame({
        'RSI': [25, 26, 27, 28, 29],
        'MACD': [1.5, 1.4, 1.3, 1.2, 1.1],
        'Signal': [1.0, 1.0, 1.0, 1.0, 1.0],
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
    strategy = HybridStrategy(model=mock_model, scaler=mock_scaler, rsi_buy=30)

    data = pd.DataFrame({
        'RSI': [25, 26, 27, 28, 29],
        'MACD': [1.5, 1.4, 1.3, 1.2, 1.1],
        'Signal': [1.0, 1.0, 1.0, 1.0, 1.0],
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
    mock_model.predict.return_value = np.array([0])  # Prédiction de baisse
    
    strategy = HybridStrategy(model=mock_model)
    
    # Données avec conditions techniques de vente
    data = pd.DataFrame({
        'RSI': [70],
        # Autres colonnes...
    })
    
    signals = strategy.generate_signals(data)
    assert signals.iloc[0] == 'SELL'

def test_hybrid_strategy_sell_with_ml_prediction():
    """Teste un signal de vente déclenché uniquement par le ML."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])  # Prédiction de baisse
    
    strategy = HybridStrategy(model=mock_model)
    
    # Données sans conditions techniques de vente
    data = pd.DataFrame({
        'RSI': [50],  # En dessous du seuil de vente
        # Autres colonnes normales...
    })
    
    signals = strategy.generate_signals(data)
    assert signals.iloc[0] == 'SELL'