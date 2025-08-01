import pytest
import pandas as pd
import numpy as np
from src.features.technical import calculate_indicators

def test_calculate_indicators():
    """Teste le calcul des indicateurs techniques sur des données simples."""
    # Créer des données de test
    data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [95, 96, 97, 98, 99],
        'Close': [100, 102, 101, 103, 105],
        'Volume': [1000, 2000, 1500, 2500, 3000]
    }, index=pd.date_range('2023-01-01', periods=5))
    
    # Calculer les indicateurs
    result = calculate_indicators(data)
    
    # Vérifier que les colonnes d'indicateurs sont présentes
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns
    assert 'Signal' in result.columns
    assert 'UpperBand' in result.columns
    assert 'LowerBand' in result.columns
    assert 'EMA50' in result.columns
    assert 'EMA200' in result.columns
    assert 'VolMA20' in result.columns
    assert 'ATR' in result.columns
    
    # Vérifier des valeurs spécifiques
    assert not pd.isna(result['RSI'].iloc[-1])
    assert result['MACD'].iloc[-1] != 0
    assert result['UpperBand'].iloc[-1] > result['LowerBand'].iloc[-1]

def test_rsi_calculation():
    """Teste spécifiquement le calcul du RSI."""
    # Cas simple: 14 périodes de hausse
    data = pd.DataFrame({
        'Close': [100 + i for i in range(15)]
    })
    
    result = calculate_indicators(data)
    
    # Après 14 périodes de hausse, RSI devrait être proche de 100
    assert result['RSI'].iloc[-1] > 90
    
    # Cas simple: 14 périodes de baisse
    data = pd.DataFrame({
        'Close': [100 - i for i in range(15)]
    })
    
    result = calculate_indicators(data)
    # Après 14 périodes de baisse, RSI devrait être proche de 0
    assert result['RSI'].iloc[-1] < 10

def test_bollinger_bands():
    """Teste le calcul des bandes de Bollinger."""
    # Données avec volatilité constante
    data = pd.DataFrame({
        'Close': [100, 102, 98, 101, 99, 103, 97, 102, 100, 101] * 3
    })
    
    result = calculate_indicators(data)
    
    # Vérifier que les bandes entourent le prix
    assert all(result['LowerBand'] < result['Close'])
    assert all(result['UpperBand'] > result['Close'])
    
    # Vérifier l'écart type
    std = result['Close'].rolling(20).std().iloc[-1]
    assert abs(result['UpperBand'].iloc[-1] - result['MA20'].iloc[-1] - 2 * std) < 0.01