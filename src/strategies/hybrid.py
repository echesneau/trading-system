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
        required_cols = {'RSI', 'MACD', 'Signal', 'Close', 'BB_Lower',
                         'BB_Upper', 'EMA_50', 'EMA_200', 'VolMA20'}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Colonnes manquantes: {missing}")

        # Préparation des features
        features = data[[
            'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower',
            'ATR', 'EMA_50', 'EMA_200', 'VolMA20'
        ]].iloc[[-1]]

        # Normalisation
        features_scaled = self.scaler.transform(features)

        # Prédiction ML
        proba = self.model.predict_proba(features_scaled)[0][1]
        prediction = 1 if proba > 0.65 else 0

        # Dernières valeurs des indicateurs
        last = data.iloc[-1]

        buy_condition = (
                (last['RSI'] < self.rsi_buy) and
                (last['MACD'] > last['Signal']) and
                (last['Close'] < last['BB_Lower']) and
                (prediction == 1)
        )

        # Conditions de vente
        sell_condition = (
                (last['RSI'] > self.rsi_sell) and
                (prediction == 0)
        )
        # Génération du signal
        if buy_condition:
            return pd.Series(['BUY'], index=[data.index[-1]])
        elif sell_condition:
            return pd.Series(['SELL'], index=[data.index[-1]])
        else:
            return pd.Series(['HOLD'], index=[data.index[-1]])

    def get_parameters(self) -> dict:
        return {
            'rsi_buy': self.rsi_buy,
            'rsi_sell': self.rsi_sell
        }

    def set_parameters(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
