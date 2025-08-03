# src/strategies/hybrid.py
import pandas as pd
import numpy as np
from .core import BaseStrategy  # Import relatif


class HybridStrategy(BaseStrategy):
    """Stratégie hybride combinant indicateurs techniques et machine learning."""

    def __init__(self, model, scaler, **kwargs):
        super().__init__()  # Appel au constructeur parent
        self.model = model
        self.scaler = scaler
        # Paramètres par défaut
        self.rsi_buy = kwargs.get('rsi_buy', 30.0)
        self.rsi_sell = kwargs.get('rsi_sell', 70.0)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calcul des indicateurs
        data = calculate_indicators(data)

        # Préparation des features
        features = data[[
            'RSI', 'MACD', 'Signal', 'UpperBand', 'LowerBand',
            'ATR', 'EMA50', 'EMA200', 'VolMA20'
        ]].iloc[[-1]]

        # Normalisation
        features_scaled = self.scaler.transform(features)

        # Prédiction ML
        proba = self.model.predict_proba(features_scaled)[0][1]
        prediction = 1 if proba > 0.65 else 0

        # Règles de décision
        if data['RSI'].iloc[-1] < 30 and prediction == 1:
            return 'BUY'
        elif data['RSI'].iloc[-1] > 70 or prediction == 0:
            return 'SELL'
        return 'HOLD'

    def get_parameters(self) -> dict:
        return {
            'rsi_buy': self.rsi_buy,
            'rsi_sell': self.rsi_sell
        }

    def set_parameters(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
