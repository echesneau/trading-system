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
        features = data[['RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower',
                         'ATR', 'EMA_50', 'EMA_200', 'VolMA20']]

        # Normalisation
        features_scaled = self.scaler.transform(features)

        # Prédiction ML
        proba = self.model.predict_proba(features_scaled)[:, 1]
        predictions = (proba > 0.65).astype(int)  # 1 si achat, 0 sinon

        # Initialisation des signaux
        signals = pd.Series('HOLD', index=data.index)

        # Conditions d'achat (vectorisées)
        buy_condition = (
                (data['RSI'] < self.rsi_buy) &
                (data['MACD'] > data['Signal']) &
                (data['Close'] < data['BB_Lower']) &
                (predictions == 1)
        )

        # Conditions de vente
        sell_condition = (
                (data['RSI'] > self.rsi_sell) &
                (predictions == 0)
        )

        signals[buy_condition] = 'BUY'
        signals[sell_condition] = 'SELL'

        return signals

    def get_parameters(self) -> dict:
        return {
            'rsi_buy': self.rsi_buy,
            'rsi_sell': self.rsi_sell
        }

    def set_parameters(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
