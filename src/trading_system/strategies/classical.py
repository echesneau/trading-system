# src/strategies/classical.py
import pandas as pd
import numpy as np
from .core import BaseStrategy  # Import relatif depuis le même package


class ClassicalStrategy(BaseStrategy):  # Hérite de BaseStrategy
    """Stratégie de trading basée sur des indicateurs techniques classiques."""

    def __init__(self, rsi_window: int = 14, rsi_buy: float = 30.0, rsi_sell: float = 70.0,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 bollinger_window: int = 20, bollinger_std: float = 2.0, **kwargs):
        super().__init__()  # Appel au constructeur parent
        self.rsi_window = rsi_window
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Génère les signaux de trading (BUY, SELL, HOLD).

        Args:
            data: DataFrame contenant les prix et indicateurs

        Returns:
            Series avec les signaux
        """
        signals = pd.Series('HOLD', index=data.index)

        # Conditions d'achat
        buy_condition = (
                (data['RSI'] < self.rsi_buy) &
                (data['MACD'] > data['MACD_Signal']) &
                (data['Close'] <= data['BB_Lower'])
        )

        # Conditions de vente
        sell_condition = (
                (data['RSI'] > self.rsi_sell) |
                (data['MACD'] < data['MACD_Signal']) &
                (data['Close'] >= data['BB_Upper'])
        )

        signals[buy_condition] = 'BUY'
        signals[sell_condition] = 'SELL'

        return signals

    def get_parameters(self) -> dict:
        """Retourne les paramètres actuels de la stratégie."""
        return {
            'rsi_window': self.rsi_window,
            'rsi_buy': self.rsi_buy,
            'rsi_sell': self.rsi_sell,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bollinger_window': self.bollinger_window,
            'bollinger_std': self.bollinger_std
        }

    def set_parameters(self, params: dict):
        """Met à jour les paramètres de la stratégie."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)