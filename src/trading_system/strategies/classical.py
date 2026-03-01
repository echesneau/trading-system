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
        close = data['Close'].values
        rsi = data['RSI'].values
        macd = data['MACD'].values
        macd_signal = data['MACD_Signal'].values
        bb_lower = data['BB_Lower'].values
        bb_upper = data['BB_Upper'].values
        adx = data['ADX'].values
        stochastic_k = data['Stochastic_%K'].values
        stochastic_d = data['Stochastic_%D'].values
        atr = data['ATR'].values
        n = len(data)
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
                return np.full(n, default, dtype=bool)
            return (k < d) & (k > stock_max)

        buy = (
                cond_lt(rsi, self.rsi_buy, default=True) &
                cond_gt(macd, macd_signal, default=True) &
                cond_lt(close, bb_lower, default=True) &
                cond_gt(adx, self.adx_min, default=True) &
                cond_gt(stochastic_k, stochastic_d, default=True) &
                cond_lt(stochastic_k, self.stock_min, default=True)
        )
        sell = (
                cond_gt(rsi, self.rsi_sell, default=False) |
                (
                        cond_lt(macd, macd_signal, default=True) &
                        cond_gt(close, bb_upper, default=True)
                ) |
                cond_stochastic_sell(stochastic_k,
                                     stochastic_d,
                                     self.stock_max,
                                     default=False
                                     )
                |
                cond_gt(atr / close, self.atr_max, default=False)
        )

        signals = np.zeros(n, dtype=np.int8)  # 0=HOLD, 1=BUY, -1=SELL
        signals[buy] = 1
        signals[sell] = -1

        return pd.Series(signals, index=data.index)

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