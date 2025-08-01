import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.strategies.hybrid import HybridStrategy

def test_hybrid_strategy_buy_signal():
    """Teste un signal d'achat avec la stratégie hybride."""
    # Créer un mock pour le modèle ML
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% de probabilité de hausse
    
    strategy = HybridStrategy(model=mock_model)
    strategy.scaler = MagicMock()
    strategy.scaler.transform.return_value = np.array([[30, 1, 0.5, 100, 90, 5, 105, 100, 1500]])
    
    # Données avec conditions techniques d'achat
    data = pd.DataFrame({
        'RSI': [35, 34, 33, 32, 31],
        'MACD': [0.5, 0.4, 0.3, 0.2, 0.1],
        'Signal': [0.6, 0.5, 0.4, 0.3, 0.2],
        'Close': [95, 94, 93, 92, 91],
        'LowerBand': [96, 95, 94, 93, 92],
        'ATR': [3, 3, 3, 3, 3],
        'EMA50': [100, 99, 98, 97, 96],
        'EMA200': [105, 104, 103, 102, 101],
        'VolMA20': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Générer les signaux
    signals = strategy.generate_signals(data)
    
    # Vérifier le signal d'achat
    assert signals.iloc[-1] == 'BUY'
    mock_model.predict_proba.assert_called_once()

def test_hybrid_strategy_no_buy_with_low_ml_confidence():
    """Vérifie qu'aucun achat n'est déclenché avec une faible confiance ML."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])  # Seulement 50% de confiance
    
    strategy = HybridStrategy(model=mock_model)
    strategy.scaler = MagicMock()
    strategy.scaler.transform.return_value = np.array([[30, 1, 0.5, 100, 90, 5, 105, 100, 1500]])
    
    data = pd.DataFrame({
        'RSI': [30],
        'MACD': [0.5],
        'Signal': [0.4],
        'Close': [91],
        'LowerBand': [90],
        # Autres colonnes nécessaires...
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