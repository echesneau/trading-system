# src/strategies/hybrid.py
import pandas as pd
import numpy as np
from .core import BaseStrategy  # Import relatif


class HybridStrategy(BaseStrategy):
    """Stratégie hybride combinant indicateurs techniques et machine learning."""

    def __init__(self, model_artifacts: dict, **kwargs):
        super().__init__()  # Appel au constructeur parent
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.feature_names = model_artifacts['feature_names']
        self._validate_features()
        # Paramètres par défaut
        self.ml_threshold = kwargs.get('ml_threshold', 0.65)
        self.rsi_buy = kwargs.get('rsi_buy', 30.0)
        self.rsi_sell = kwargs.get('rsi_sell', 70.0)

        self.required_cols = self._determine_required_columns()

    def _determine_required_columns(self) -> set:
        """Détermine dynamiquement les colonnes requises"""
        required = set()

        # 1. Features du modèle ML
        required.update(self.feature_names)

        # 2. Colonnes pour les conditions de trading
        required.update(['RSI', 'MACD', 'MACD_Signal', 'Close', 'BB_Lower'])  # Toujours nécessaires

        # 3. Colonnes conditionnelles selon les paramètres
        if hasattr(self, 'rsi_buy') or hasattr(self, 'rsi_sell'):
            required.add('RSI')


        return required

    def _validate_features(self):
        required = {'RSI', 'MACD', 'BB_Upper', 'BB_Lower'}
        missing = required - set(self.feature_names)
        if missing:
            raise ValueError(f"Features manquantes dans le modèle: {missing}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calcul des indicateurs
        available_cols = set(data.columns)
        missing_cols = self.required_cols - available_cols
        if missing_cols:
            raise ValueError(
                f"Colonnes manquantes: {missing_cols}. "
                f"Disponibles: {list(available_cols)[:10]}..."
            )

        # Préparation des features
        features = data[list(self.required_cols.intersection(available_cols))]

        # Normalisation
        features_scaled = self.scaler.transform(data[self.feature_names])

        # Prédiction ML
        proba = self.model.predict_proba(features_scaled)[:, 1]
        predictions = (proba > self.ml_threshold).astype(int)  # 1 si achat, 0 sinon

        # Initialisation des signaux
        signals = pd.Series('HOLD', index=data.index)

        # Conditions d'achat (vectorisées)
        buy_condition = (
                (data['RSI'] < self.rsi_buy) &
                (data['MACD'] > data['MACD_Signal']) &
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
