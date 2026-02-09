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

    def generate_signals(self, data: pd.DataFrame, to_str: bool = False) -> pd.Series:
        """
        Génère les signaux de trading (BUY, SELL, HOLD).

        Args:
            data: DataFrame contenant les prix et indicateurs
            to_str: bool
                to transform int to str (0=HOLD, 1=BUY, -1=SELL) to ("HOLD", "BUY", "SELL")

        Returns:
            Series avec les signaux
        """
        buy = (
                (data['RSI'].values < self.rsi_buy) &
                (data['MACD'].values > data['MACD_Signal'].values) &
                (data['Close'].values <= data['BB_Lower'].values)
        )
        sell = (
                (data['RSI'].values > self.rsi_sell) |
                (
                        (data['MACD'].values < data['MACD_Signal'].values) &
                        (data['Close'].values >= data['BB_Upper'].values)
                )
        )
        signals = np.zeros(len(data), dtype=np.int8)  # 0=HOLD, 1=BUY, -1=SELL
        signals[buy] = 1
        signals[sell] = -1

        signals = pd.Series(signals, index=data.index)
        if to_str:
            signals = signals.map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
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