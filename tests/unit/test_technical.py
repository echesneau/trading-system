# tests/unit/test_technical_indicators.py
from platform import machine

import pytest
import pandas as pd
import numpy as np
from src.features.technical import calculate_indicators, calculate_price_volume_trend


def test_calculate_indicators_basic():
    """Teste le calcul des indicateurs avec des données basiques."""
    # Créer des données de test
    data = pd.DataFrame({
        'Open': [100 + i for i in range(30)],
        'High': [105 + i for i in range(30)],
        'Low': [95 + i for i in range(30)],
        'Close': [100 + i for i in range(30)],
        'Volume': [1000 + i * 100 for i in range(30)]
    }, index=pd.date_range('2023-01-01', periods=30))

    # Calculer les indicateurs
    result = calculate_indicators(data, ema_windows=[5, 10, 20], adx_window=7, )

    # Vérifier les colonnes ajoutées
    expected_indicators = [
        'RSI', 'MACD', 'MACD_Signal', 'Stochastic_%K', 'Stochastic_%D',
        'ATR', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'EMA_5', 'EMA_10', 'EMA_20', 'ADX',
        'OBV', 'Volume_MA_20', 'Price_Volume_Trend', 'Daily_Return'
    ]

    for indicator in expected_indicators:
        assert indicator in result.columns, f"{indicator} manquant dans les résultats"

    # Vérifier que les valeurs sont calculées
    assert not result['RSI'].isna().all()
    assert not result['MACD'].isna().all()
    assert not result['ATR'].isna().all()
    assert result['BB_Upper'].iloc[-1] > result['BB_Lower'].iloc[-1]
    assert result['MACD'].iloc[-1] != 0


def test_rsi_calculation():
    """Teste spécifiquement le calcul du RSI."""
    # Cas simple: 14 périodes de hausse
    data = pd.DataFrame({
        'Open': [100 + i for i in range(30)],
        'High': [105 + i for i in range(30)],
        'Low': [95 + i for i in range(30)],
        'Close': [100 + i for i in range(30)],
        'Volume': [1000 + i * 100 for i in range(30)]
    }, index=pd.date_range('2023-01-01', periods=30))

    result = calculate_indicators(data, ema_windows=[])

    # Après 14 périodes de hausse, RSI devrait être proche de 100
    assert result['RSI'].iloc[-1] > 90

    # Cas simple: 14 périodes de baisse
    data = pd.DataFrame({
        'Open': [100 + i for i in range(30)],
        'High': [105 + i for i in range(30)],
        'Low': [95 + i for i in range(30)],
        'Close': [100 - i for i in range(30)],
        'Volume': [1000 + i * 100 for i in range(30)]
    }, index=pd.date_range('2023-01-01', periods=30))

    result = calculate_indicators(data,ema_windows=[])
    # Après 14 périodes de baisse, RSI devrait être proche de 0
    assert result['RSI'].iloc[-1] < 10

def test_missing_data_handling():
    """Teste la gestion des données manquantes."""
    # Données avec colonnes manquantes
    bad_data = pd.DataFrame({
        'Open': [100, 101],
        'Close': [100, 102]
    })

    with pytest.raises(ValueError) as excinfo:
        calculate_indicators(bad_data)
    assert "Données manquantes" in str(excinfo.value)


def test_indicator_calculation():
    """Teste des calculs spécifiques d'indicateurs."""
    # Données avec tendance haussière constante
    data = pd.DataFrame({
        'Open': [100 + i for i in range(30)],
        'High': [105 + i for i in range(30)],
        'Low': [95 + i for i in range(30)],
        'Close': [100 + i for i in range(30)],
        'Volume': [1000] * 30
    })

    result = calculate_indicators(data, ema_windows=[5, 10, 20], adx_window=7)

    # RSI devrait être élevé pour une tendance haussière
    assert result['RSI'].iloc[-1] > 70

    # MACD devrait être positif
    assert result['MACD'].iloc[-1] > 0

    # EMA20 < EMA50 < EMA200 dans une tendance
    assert result['EMA_5'].iloc[-1] > result['EMA_10'].iloc[-1]
    assert result['EMA_10'].iloc[-1] > result['EMA_20'].iloc[-1]


def test_custom_indicators():
    """Teste les indicateurs personnalisés."""
    data = pd.DataFrame({
        'Open': [100, 101, 102, 101, 100],
        'High': [105, 106, 107, 106, 105],
        'Low': [95, 96, 97, 96, 95],
        'Close': [100, 102, 103, 101, 100],
        'Volume': [1000, 2000, 3000, 2000, 1000]
    })

    result = calculate_price_volume_trend(data)

    # Price_Volume_Trend devrait être positif quand prix et volume montent
    assert result.iloc[2] > result.iloc[1]

    # Et négatif quand ils baissent
    assert result.iloc[4] < result.iloc[3]


def test_bollinger_bands():
    """Teste le calcul des bandes de Bollinger."""
    # Données avec volatilité constante
    data = pd.DataFrame({
        'Open': [100 + i for i in range(30)],
        'High': [105 + i for i in range(30)],
        'Low': [95 + i for i in range(30)],
        'Close': [100, 102, 98, 101, 99, 103, 97, 102, 100, 101] * 3,
        'Volume': [1000 + i * 100 for i in range(30)]
    }, index=pd.date_range('2023-01-01', periods=30))


    result = calculate_indicators(data, ema_windows=[20])

    # Vérifier que les bandes entourent le prix
    not_null_result = result[pd.notnull(result['BB_Lower'])]
    assert all(not_null_result['BB_Lower'] < not_null_result['Close'])
    assert all(not_null_result['BB_Upper'] > not_null_result['Close'])
    # Vérifier l'écart type
    # std = result['Close'].rolling(20).std().iloc[-1]
    # assert abs(not_null_result['BB_Upper'].iloc[-1] - not_null_result['EMA_20'].iloc[-1] - 2 * std) < 0.01