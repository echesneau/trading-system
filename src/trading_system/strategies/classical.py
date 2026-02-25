# src/strategies/classical.py
import pandas as pd
import numpy as np
from .core import BaseStrategy  # Import relatif depuis le même package


class ClassicalStrategy(BaseStrategy):  # Hérite de BaseStrategy
    """Stratégie de trading basée sur des indicateurs techniques classiques."""

    def __init__(self, rsi_window: int = 14, rsi_buy: float = 30.0, rsi_sell: float = 70.0,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 bollinger_window: int = 20, bollinger_std: float = 2.0,
                 adx_min: int = None, stock_min: int = None, stock_max: int = None, atr_max: int = None,
                 **kwargs):
        super().__init__()  # Appel au constructeur parent
        self.rsi_window = rsi_window
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.adx_min = adx_min
        self.stock_min = stock_min
        self.stock_max = stock_max
        self.atr_max = atr_max

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Génère les signaux de trading (BUY, SELL, HOLD).

        Args:
            data: DataFrame contenant les prix et indicateurs
            to_str: bool
                to transform int to str (0=HOLD, 1=BUY, -1=SELL) to ("HOLD", "BUY", "SELL")

        Returns:
            Series avec les signaux
        """
        def cond_lt(arr, threshold, default=True):
            if threshold is None :
                return np.full_like(arr, default, dtype=bool)
            threshold = np.asarray(threshold)
            if np.all(np.isnan(threshold)):
                return np.full_like(arr, default, dtype=bool)
            return arr < threshold

        def cond_gt(arr, threshold, default=True):
            if threshold is None:
                return np.full_like(arr, default, dtype=bool)
            threshold = np.asarray(threshold)
            if np.all(np.isnan(threshold)):
                return np.full_like(arr, default, dtype=bool)
            return arr > threshold

        def cond_stochastic_sell(k, d, stock_max, default=True):
            # Si k ou stock_max sont nan → retourne default
            if np.all(np.isnan(k)) or pd.isnull(stock_max):
                return np.full_like(k, default, dtype=bool)

            return (k < d) & (k > stock_max)

        buy = (
                cond_lt(data['RSI'].values, self.rsi_buy, default=True) &
                cond_gt(data['MACD'].values, data['MACD_Signal'].values, default=True) &
                cond_lt(data['Close'].values, data['BB_Lower'].values, default=True) &
                cond_gt(data["ADX"].values, self.adx_min, default=True) &
                cond_gt(data["Stochastic_%K"].values, data["Stochastic_%D"].values, default=True) &
                cond_lt(data["Stochastic_%K"], self.stock_min, default=True)
        )
        sell = (
                cond_gt(data['RSI'].values, self.rsi_sell, default=False) |
                (
                        cond_lt(data['MACD'].values, data['MACD_Signal'].values, default=True) &
                        cond_gt(data['Close'].values, data['BB_Upper'].values, default=True)
                ) |
                cond_stochastic_sell(data["Stochastic_%K"].values,
                                     data["Stochastic_%D"].values,
                                     self.stock_max,
                                     default=False
                                     )
                |
                cond_gt(data["ATR"].values / data["Close"].values, self.atr_max, default=False)
        )

        signals = np.zeros(len(data), dtype=np.int8)  # 0=HOLD, 1=BUY, -1=SELL
        signals[buy] = 1
        signals[sell] = -1

        signals = pd.Series(signals, index=data.index)
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